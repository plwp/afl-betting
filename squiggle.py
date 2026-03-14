"""Fetch historical predictions from the Squiggle API (wisdom of crowds).

Also provides enhanced signals: top-model predictions and model disagreement.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd

from config import DATA_DIR


SQUIGGLE_API = "https://api.squiggle.com.au/"
SQUIGGLE_CACHE = os.path.join(DATA_DIR, "squiggle_cache")
HEADERS = {"User-Agent": "afl-tipping-bot/1.0 (github.com/plwp/afl-betting)"}
STANDINGS_CACHE_TTL = 86400  # 24 hours


def _cache_path(year: int) -> str:
    os.makedirs(SQUIGGLE_CACHE, exist_ok=True)
    return os.path.join(SQUIGGLE_CACHE, f"tips_{year}.json")


def fetch_squiggle_tips(year: int) -> list:
    """Fetch all model predictions for a given year from Squiggle."""
    cache_file = _cache_path(year)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Fetching Squiggle tips for {year}...")
    params = {"q": "tips", "year": year}
    resp = requests.get(SQUIGGLE_API, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("tips", [])

    with open(cache_file, "w") as f:
        json.dump(data, f)

    time.sleep(1)  # Be polite to Squiggle API
    return data


def build_squiggle_consensus(years: range) -> pd.DataFrame:
    """Build a DataFrame with consensus prediction for each match.

    Returns DataFrame with columns: year, round, home_team, away_team, squiggle_prob_home.
    squiggle_prob_home is the average predicted home win probability across all Squiggle models.
    """
    all_tips = []
    for year in years:
        tips = fetch_squiggle_tips(year)
        all_tips.extend(tips)

    if not all_tips:
        return pd.DataFrame()

    df = pd.DataFrame(all_tips)

    # Squiggle tips have: hteam, ateam, year, round, confidence, hconfidence, tip, source
    # 'hconfidence' is the home team win probability (0-100) from each model
    # We want the average across all models for each match

    if "hconfidence" not in df.columns or "hteam" not in df.columns:
        return pd.DataFrame()

    # Some models don't provide hconfidence
    df = df.dropna(subset=["hconfidence"])
    df["hconfidence"] = pd.to_numeric(df["hconfidence"], errors="coerce")
    df = df.dropna(subset=["hconfidence"])

    # Normalize team names
    from config import TEAM_NAME_MAP
    df["home_team"] = df["hteam"].map(lambda n: TEAM_NAME_MAP.get(n, n))
    df["away_team"] = df["ateam"].map(lambda n: TEAM_NAME_MAP.get(n, n))

    # Average across all models per match
    consensus = (
        df.groupby(["year", "round", "home_team", "away_team"])["hconfidence"]
        .mean()
        .reset_index()
    )
    consensus["squiggle_prob_home"] = consensus["hconfidence"] / 100.0
    consensus = consensus.rename(columns={"round": "round_num"})

    # Convert round_num to match our format
    consensus["round_num"] = consensus["round_num"].astype(str)
    consensus["year"] = consensus["year"].astype(int)

    return consensus[["year", "round_num", "home_team", "away_team", "squiggle_prob_home"]]


# ---------------------------------------------------------------------------
# Enhanced Squiggle: top-model predictions + model disagreement
# ---------------------------------------------------------------------------

def _standings_cache_path(year: int) -> str:
    os.makedirs(SQUIGGLE_CACHE, exist_ok=True)
    return os.path.join(SQUIGGLE_CACHE, f"standings_{year}.json")


def fetch_standings(year: int) -> list:
    """Fetch model standings for a given year from Squiggle."""
    cache_file = _standings_cache_path(year)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < STANDINGS_CACHE_TTL:
            with open(cache_file) as f:
                return json.load(f)

    print(f"  Fetching Squiggle standings for {year}...")
    params = {"q": "standings", "year": year}
    resp = requests.get(SQUIGGLE_API, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("standings", [])

    with open(cache_file, "w") as f:
        json.dump(data, f)

    time.sleep(1)
    return data


def get_top_models(year: int, n: int = 3) -> list:
    """Return source names of the top N models by accuracy for a given year."""
    standings = fetch_standings(year)
    if not standings:
        return []

    # Sort by 'pct' (tipping accuracy percentage) descending
    ranked = sorted(standings, key=lambda s: s.get("pct", 0), reverse=True)
    return [s["source"] for s in ranked[:n] if "source" in s]


def fetch_current_round_tips(year: int, round_num: int) -> list:
    """Fetch tips for a specific round (used for live scanning)."""
    cache_key = f"tips_{year}_r{round_num}"
    cache_file = os.path.join(SQUIGGLE_CACHE, f"{cache_key}.json")
    os.makedirs(SQUIGGLE_CACHE, exist_ok=True)

    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 900:  # 15 min cache for live tips
            with open(cache_file) as f:
                return json.load(f)

    print(f"  Fetching Squiggle tips for {year} round {round_num}...")
    params = {"q": "tips", "year": year, "round": round_num}
    resp = requests.get(SQUIGGLE_API, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("tips", [])

    with open(cache_file, "w") as f:
        json.dump(data, f)

    time.sleep(1)
    return data


def get_enhanced_squiggle_data(year: int, round_num: int) -> dict:
    """Compute enhanced Squiggle signals for current round matches.

    Returns dict keyed by (home_team, away_team) with:
      - squiggle_top3_prob: avg home win prob from top 3 models
      - squiggle_model_spread: std dev of home win prob across all models
    """
    from config import TEAM_NAME_MAP

    top_models = get_top_models(year)
    tips = fetch_current_round_tips(year, round_num)

    if not tips:
        return {}

    df = pd.DataFrame(tips)
    if "hconfidence" not in df.columns or "hteam" not in df.columns:
        return {}

    df = df.dropna(subset=["hconfidence"])
    df["hconfidence"] = pd.to_numeric(df["hconfidence"], errors="coerce")
    df = df.dropna(subset=["hconfidence"])
    df["home_team"] = df["hteam"].map(lambda n: TEAM_NAME_MAP.get(n, n))
    df["away_team"] = df["ateam"].map(lambda n: TEAM_NAME_MAP.get(n, n))
    df["prob"] = df["hconfidence"] / 100.0

    result = {}
    for (home, away), group in df.groupby(["home_team", "away_team"]):
        # Top 3 model average
        if top_models:
            top3 = group[group["source"].isin(top_models)]
            top3_prob = float(top3["prob"].mean()) if not top3.empty else float(group["prob"].mean())
        else:
            top3_prob = float(group["prob"].mean())

        # Model disagreement (std dev across all models)
        model_spread = float(group["prob"].std()) if len(group) > 1 else 0.0

        result[(home, away)] = {
            "squiggle_top3_prob": top3_prob,
            "squiggle_model_spread": model_spread,
        }

    if result:
        print(f"Squiggle enhanced: {len(result)} matches, top models: {top_models[:3]}")

    return result
