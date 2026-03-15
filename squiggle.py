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
FINAL_ROUND_MAP = {
    "Elimination Final": "25",
    "Qualifying Final": "25",
    "Semi Final": "26",
    "Preliminary Final": "27",
    "Grand Final": "28",
}


def _normalize_round_id(value) -> str:
    """Normalize round identifiers across data sources."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in FINAL_ROUND_MAP:
        return FINAL_ROUND_MAP[text]
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _prepare_tips_df(tips: list, require_correct: bool = False) -> pd.DataFrame:
    """Normalize Squiggle tips into a typed DataFrame."""
    if not tips:
        return pd.DataFrame()

    df = pd.DataFrame(tips)
    required = {"hconfidence", "hteam", "ateam", "source"}
    if require_correct:
        required.add("correct")
    if not required.issubset(df.columns):
        return pd.DataFrame()

    from config import TEAM_NAME_MAP

    df = df.copy()
    df["hconfidence"] = pd.to_numeric(df["hconfidence"], errors="coerce")
    df = df.dropna(subset=["hconfidence"])
    if require_correct:
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        df = df.dropna(subset=["correct"])

    exclude = {"Aggregate", "Punters"}
    df = df[~df["source"].isin(exclude)]
    df["home_team"] = df["hteam"].map(lambda n: TEAM_NAME_MAP.get(n, n))
    df["away_team"] = df["ateam"].map(lambda n: TEAM_NAME_MAP.get(n, n))
    df["prob"] = df["hconfidence"] / 100.0
    df["round_num"] = df.get("round", "").map(_normalize_round_id)
    return df


def _top_models_from_df(df: pd.DataFrame, n: int = 3) -> list[str]:
    """Rank models by accuracy using only rows already observed."""
    if df.empty or "correct" not in df.columns:
        return []
    accuracy = df.groupby("source")["correct"].mean().sort_values(ascending=False)
    return list(accuracy.head(n).index)


def _build_enhanced_round_features(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Build leakage-safe enhanced Squiggle features round by round."""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["round_sort"] = pd.to_numeric(df["round_num"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    round_keys = (
        df[["round_num", "round_sort"] + (["date"] if "date" in df.columns else [])]
        .drop_duplicates("round_num")
        .sort_values(
            ["round_sort"] + (["date"] if "date" in df.columns else []),
            kind="mergesort",
            na_position="last",
        )["round_num"]
        .tolist()
    )

    rows = []
    history = df.iloc[0:0].copy()
    for round_num in round_keys:
        current = df[df["round_num"] == round_num].copy()
        top_models = _top_models_from_df(history, n=top_n)

        for (home, away), group in current.groupby(["home_team", "away_team"], sort=False):
            if top_models:
                top_group = group[group["source"].isin(top_models)]
                top_prob = float(top_group["prob"].mean()) if not top_group.empty else float(group["prob"].mean())
            else:
                top_prob = float(group["prob"].mean())

            rows.append({
                "year": int(group["year"].iloc[0]),
                "round_num": round_num,
                "home_team": home,
                "away_team": away,
                "squiggle_top3_prob": top_prob,
                "squiggle_model_spread": float(group["prob"].std()) if len(group) > 1 else 0.0,
            })

        history = pd.concat([history, current], ignore_index=True)

    return pd.DataFrame(rows)


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

    df = _prepare_tips_df(all_tips, require_correct=False)
    if df.empty:
        return pd.DataFrame()

    # Average across all models per match
    consensus = (
        df.groupby(["year", "round_num", "home_team", "away_team"])["hconfidence"]
        .mean()
        .reset_index()
    )
    consensus["squiggle_prob_home"] = consensus["hconfidence"] / 100.0
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
    """Return source names of the top N models by accuracy for a given year.

    Computes accuracy from the tips data (correct column) rather than
    relying on a separate standings endpoint.
    """
    return get_top_models_up_to_round(year, n=n)


def get_top_models_up_to_round(year: int, n: int = 3, max_round=None) -> list:
    """Return top models using only completed rounds before max_round."""
    tips = fetch_squiggle_tips(year)
    df = _prepare_tips_df(tips, require_correct=True)
    if df.empty:
        return []

    if max_round is not None:
        max_round_key = _normalize_round_id(max_round)
        current_round_sort = pd.to_numeric(pd.Series([max_round_key]), errors="coerce").iloc[0]
        df["round_sort"] = pd.to_numeric(df["round_num"], errors="coerce")
        df = df[df["round_sort"] < current_round_sort]

    return _top_models_from_df(df, n=n)


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

    top_models = get_top_models_up_to_round(year, max_round=round_num)
    tips = fetch_current_round_tips(year, round_num)

    if not tips:
        return {}

    df = _prepare_tips_df(tips, require_correct=False)
    if df.empty:
        return {}

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


def build_enhanced_squiggle_historical(years: range) -> pd.DataFrame:
    """Build enhanced Squiggle features for historical data.

    For each year, identifies top 3 models by accuracy, then computes
    per-match squiggle_top3_prob and squiggle_model_spread.

    Returns DataFrame with columns:
      year, round_num, home_team, away_team, squiggle_top3_prob, squiggle_model_spread
    """
    all_rows = []

    for year in years:
        tips = fetch_squiggle_tips(year)
        year_df = _prepare_tips_df(tips, require_correct=True)
        if year_df.empty:
            continue

        all_rows.append(_build_enhanced_round_features(year_df, top_n=3))

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    print(f"  Enhanced Squiggle: {len(result)} matches across {len(years)} years")
    return result
