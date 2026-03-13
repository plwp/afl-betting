"""Live odds scanner using The Odds API."""

import os
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_CACHE_DIR,
    ODDS_CACHE_TTL, AU_BOOKMAKERS, EDGE_THRESHOLD,
    FEATURE_COLS, TEAM_NAME_MAP,
)
from sizing import kelly_stake, edge


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

    # Check cache
    if not force_refresh and os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < ODDS_CACHE_TTL:
            with open(cache_file) as f:
                cached = json.load(f)
            print(f"Using cached odds ({int(time.time() - mtime)}s old)")
            return cached["data"]

    # Fetch from API
    print("Fetching live odds from The Odds API...")
    resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  {len(data)} events, {remaining} API requests remaining")

    # Cache response
    with open(cache_file, "w") as f:
        json.dump({"timestamp": time.time(), "data": data}, f)

    return data


def _normalize_api_team(name: str) -> str:
    """Normalize team names from Odds API to our canonical form."""
    # The Odds API uses slightly different names
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


def scan_value_bets(odds_df: pd.DataFrame,
                    model_probs: dict,
                    bankroll: float,
                    edge_threshold: float = EDGE_THRESHOLD) -> pd.DataFrame:
    """Find +EV bets by comparing model probs to market odds.

    Args:
        odds_df: DataFrame from parse_odds().
        model_probs: dict mapping (home_team, away_team) -> home_win_prob.
        bankroll: Current bankroll for Kelly sizing.
        edge_threshold: Minimum edge to flag a bet.

    Returns:
        DataFrame of value bets.
    """
    value_bets = []

    for _, row in odds_df.iterrows():
        key = (row["home_team"], row["away_team"])
        if key not in model_probs:
            continue

        prob_home = model_probs[key]
        prob_away = 1 - prob_home

        # Home side
        if row["best_home_odds"] > 0:
            e = edge(prob_home, row["best_home_odds"])
            if e > edge_threshold:
                stake = kelly_stake(prob_home, row["best_home_odds"], bankroll)
                value_bets.append({
                    "match": f"{row['home_team']} v {row['away_team']}",
                    "side": row["home_team"],
                    "model_prob": prob_home,
                    "implied_prob": 1 / row["best_home_odds"],
                    "best_odds": row["best_home_odds"],
                    "bookmaker": row["best_home_bookie"],
                    "edge": e,
                    "kelly_stake": stake,
                    "commence": row["commence_time"],
                })

        # Away side
        if row["best_away_odds"] > 0:
            e = edge(prob_away, row["best_away_odds"])
            if e > edge_threshold:
                stake = kelly_stake(prob_away, row["best_away_odds"], bankroll)
                value_bets.append({
                    "match": f"{row['home_team']} v {row['away_team']}",
                    "side": row["away_team"],
                    "model_prob": prob_away,
                    "implied_prob": 1 / row["best_away_odds"],
                    "best_odds": row["best_away_odds"],
                    "bookmaker": row["best_away_bookie"],
                    "edge": e,
                    "kelly_stake": stake,
                    "commence": row["commence_time"],
                })

    result = pd.DataFrame(value_bets)
    if not result.empty:
        result = result.sort_values("edge", ascending=False).reset_index(drop=True)
    return result
