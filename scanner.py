"""Live odds scanner using The Odds API."""

import os
import json
import time
import hashlib
import requests
import pandas as pd
from datetime import datetime

from config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_CACHE_DIR,
    ODDS_CACHE_TTL, AU_BOOKMAKERS, EDGE_THRESHOLD,
    FAVOURITE_ONLY, MAX_ODDS, MIN_MODEL_PROB, TEAM_NAME_MAP,
    ARB_STAKE_FRACTION, MIN_STAKE,
)
from sizing import kelly_stake, edge
from strategy import DOG_MIN_FORM, DOG_MAX_SCORING_GAP


def _cache_path(params: dict) -> str:
    """Generate cache file path from request params."""
    os.makedirs(ODDS_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return os.path.join(ODDS_CACHE_DIR, f"{key}.json")


def fetch_odds(force_refresh: bool = False) -> list:
    """Fetch AFL h2h odds from The Odds API with caching."""
    if not ODDS_API_KEY or ODDS_API_KEY == "your_key_here":
        raise ValueError(
            "Set ODDS_API_KEY in .env (get free key at https://the-odds-api.com)"
        )

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "au",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    cache_file = _cache_path(params)

    if not force_refresh and os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < ODDS_CACHE_TTL:
            with open(cache_file) as f:
                cached = json.load(f)
            print(f"Using cached odds ({int(time.time() - mtime)}s old)")
            return cached["data"]

    print("Fetching live odds from The Odds API...")
    resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  {len(data)} events, {remaining} API requests remaining")

    with open(cache_file, "w") as f:
        json.dump({"timestamp": time.time(), "data": data}, f)

    return data


def _normalize_api_team(name: str) -> str:
    """Normalize team names from Odds API to our canonical form."""
    api_map = {
        "Brisbane Lions": "Brisbane",
        "Greater Western Sydney Giants": "GWS Giants",
        "GWS Giants": "GWS Giants",
        "Gold Coast Suns": "Gold Coast",
    }
    if name in api_map:
        return api_map[name]
    return TEAM_NAME_MAP.get(name, name)


def parse_odds(events: list) -> pd.DataFrame:
    """Parse API response into a DataFrame with best odds per team."""
    rows = []
    for event in events:
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = _normalize_api_team(home_raw)
        away = _normalize_api_team(away_raw)
        commence = event.get("commence_time", "")

        best_home_odds = 0
        best_away_odds = 0
        best_home_bookie = ""
        best_away_bookie = ""

        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            if AU_BOOKMAKERS and bm_key not in AU_BOOKMAKERS:
                continue
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = _normalize_api_team(outcome.get("name", ""))
                    price = outcome.get("price", 0)
                    if team == home and price > best_home_odds:
                        best_home_odds = price
                        best_home_bookie = bm_key
                    elif team == away and price > best_away_odds:
                        best_away_odds = price
                        best_away_bookie = bm_key

        rows.append({
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "best_home_odds": best_home_odds,
            "best_home_bookie": best_home_bookie,
            "best_away_odds": best_away_odds,
            "best_away_bookie": best_away_bookie,
        })

    return pd.DataFrame(rows)


def _arb_stakes(odds_home: float, odds_away: float, bankroll: float,
                 stake_fraction: float = ARB_STAKE_FRACTION) -> tuple:
    """Compute stakes for a cross-bookie arbitrage.

    Returns (stake_home, stake_away, guaranteed_profit) or None if no arb.
    """
    if odds_home <= 1 or odds_away <= 1:
        return None
    inv_sum = 1 / odds_home + 1 / odds_away
    if inv_sum >= 1:
        return None  # no arb
    total = bankroll * stake_fraction
    stake_home = round(total * (1 / odds_home) / inv_sum, 2)
    stake_away = round(total * (1 / odds_away) / inv_sum, 2)
    profit = round(total / inv_sum - total, 2)
    if stake_home < MIN_STAKE or stake_away < MIN_STAKE:
        return None
    return stake_home, stake_away, profit


def _is_hot_dog(home_team: str, away_team: str, season_form: dict) -> bool:
    """Check if a home underdog passes the hot-dog filter."""
    if not season_form:
        return False
    home_data = season_form.get(home_team)
    away_data = season_form.get(away_team)
    if not home_data or not away_data:
        return False
    scoring_gap = away_data["scoring_ewma"] - home_data["scoring_ewma"]
    return home_data["form_5"] >= DOG_MIN_FORM and scoring_gap <= DOG_MAX_SCORING_GAP


def scan_value_bets(odds_df: pd.DataFrame,
                    model_probs: dict,
                    bankroll: float,
                    edge_threshold: float = EDGE_THRESHOLD,
                    max_odds: float = MAX_ODDS,
                    min_model_prob: float = MIN_MODEL_PROB,
                    favourite_only: bool = FAVOURITE_ONLY,
                    season_form: dict = None) -> pd.DataFrame:
    """Find +EV bets and cross-bookie arbitrages.

    Arbs bypass all filters (favourite-only, odds cap, model prob).
    For non-arb EV bets, when favourite_only=True only bets the favourite side
    UNLESS the home underdog passes the hot-dog filter (hot form + close scoring).
    season_form: dict from squiggle.fetch_season_form() for hot-dog checks.
    """
    value_bets = []

    for _, row in odds_df.iterrows():
        key = (row["home_team"], row["away_team"])
        match_label = f"{row['home_team']} v {row['away_team']}"
        odds_h = row["best_home_odds"]
        odds_a = row["best_away_odds"]

        # --- Arb check first (bypasses all filters) ---
        if odds_h > 0 and odds_a > 0:
            arb = _arb_stakes(odds_h, odds_a, bankroll)
            if arb is not None:
                stake_home, stake_away, profit = arb
                arb_margin = 1 - (1 / odds_h + 1 / odds_a)
                value_bets.append({
                    "match": match_label,
                    "side": row["home_team"],
                    "model_prob": model_probs.get(key, 0),
                    "implied_prob": 1 / odds_h,
                    "best_odds": odds_h,
                    "bookmaker": row["best_home_bookie"],
                    "edge": arb_margin,
                    "kelly_stake": stake_home,
                    "commence": row["commence_time"],
                    "is_arb": True,
                    "arb_profit": profit,
                })
                value_bets.append({
                    "match": match_label,
                    "side": row["away_team"],
                    "model_prob": model_probs.get(key, 0),
                    "implied_prob": 1 / odds_a,
                    "best_odds": odds_a,
                    "bookmaker": row["best_away_bookie"],
                    "edge": arb_margin,
                    "kelly_stake": stake_away,
                    "commence": row["commence_time"],
                    "is_arb": True,
                    "arb_profit": profit,
                })
                continue  # arb found — skip EV logic for this match

        # --- Normal EV logic ---
        if key not in model_probs:
            continue

        prob_home = model_probs[key]
        prob_away = 1 - prob_home
        implied_home = 1 / odds_h if odds_h > 0 else 0
        implied_away = 1 / odds_a if odds_a > 0 else 0
        candidates = []

        for side_team, prob, implied, odds_val, bookie_col in [
            (row["home_team"], prob_home, implied_home, odds_h, "best_home_bookie"),
            (row["away_team"], prob_away, implied_away, odds_a, "best_away_bookie"),
        ]:
            if odds_val <= 0:
                continue
            is_fav = implied > 0.5

            if favourite_only:
                if is_fav:
                    if prob < min_model_prob or odds_val > max_odds:
                        continue
                else:
                    # Underdog: apply hot-dog filter (home dogs only)
                    is_home = (side_team == row["home_team"])
                    if not is_home:
                        continue
                    if not _is_hot_dog(side_team, key[1], season_form):
                        continue

            e = edge(prob, odds_val)
            if e > edge_threshold:
                stake = kelly_stake(prob, odds_val, bankroll)
                if stake > 0:
                    candidates.append({
                        "match": match_label,
                        "side": side_team,
                        "model_prob": prob,
                        "implied_prob": implied,
                        "best_odds": odds_val,
                        "bookmaker": row[bookie_col],
                        "edge": e,
                        "kelly_stake": stake,
                        "commence": row["commence_time"],
                        "is_arb": False,
                    })

        if candidates:
            value_bets.append(max(candidates, key=lambda b: (b["edge"], b["model_prob"])))

    result = pd.DataFrame(value_bets)
    if not result.empty:
        result = result.sort_values(
            ["is_arb", "edge"], ascending=[False, False],
        ).reset_index(drop=True)
    return result
