"""Fetch historical predictions from the Squiggle API (wisdom of crowds)."""

import os
import time
import json
import requests
import pandas as pd

from config import DATA_DIR


SQUIGGLE_API = "https://api.squiggle.com.au/"
SQUIGGLE_CACHE = os.path.join(DATA_DIR, "squiggle_cache")
HEADERS = {"User-Agent": "afl-tipping-bot/1.0 (github.com/plwp/afl-betting)"}


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
