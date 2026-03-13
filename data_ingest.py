"""Download and merge AFL match data with historical odds."""

import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO, StringIO
from config import (
    MATCH_CSV_URL, ODDS_XLSX_URL, MATCH_YEARS,
    DATA_DIR, MERGED_PATH, TEAM_NAME_MAP,
)


def normalize_team(name: str) -> str:
    """Map any team name variant to canonical form."""
    return TEAM_NAME_MAP.get(name, name)


def download_match_data() -> pd.DataFrame:
    """Download match CSVs from akareen/AFL-Data-Analysis repo."""
    frames = []
    for year in MATCH_YEARS:
        url = MATCH_CSV_URL.format(year=year)
        print(f"  Downloading matches {year}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)

    # Compute scores: goals * 6 + behinds
    matches["home_score"] = (
        matches["team_1_final_goals"] * 6 + matches["team_1_final_behinds"]
    )
    matches["away_score"] = (
        matches["team_2_final_goals"] * 6 + matches["team_2_final_behinds"]
    )
    matches["margin"] = matches["home_score"] - matches["away_score"]
    matches["home_win"] = (matches["margin"] > 0).astype(int)

    # Normalize team names
    matches["home_team"] = matches["team_1_team_name"].map(normalize_team)
    matches["away_team"] = matches["team_2_team_name"].map(normalize_team)

    # Parse date (format: "2023-03-16 19:20")
    matches["date"] = pd.to_datetime(
        matches["date"].str.strip(), format="mixed"
    ).dt.normalize()
    matches["year"] = matches["year"].astype(int)

    keep = [
        "date", "year", "round_num", "venue",
        "home_team", "away_team",
        "home_score", "away_score", "margin", "home_win",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]
    return matches[keep].copy()


def download_odds_data() -> pd.DataFrame:
    """Download historical odds from AusSportsBetting."""
    print("  Downloading odds data...")
    resp = requests.get(ODDS_XLSX_URL, timeout=60,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    odds = pd.read_excel(BytesIO(resp.content), header=1)
    print(f"  Odds columns: {odds.columns.tolist()[:20]}...")

    # Normalize
    odds["date"] = pd.to_datetime(odds["Date"]).dt.normalize()
    odds["home_team"] = odds["Home Team"].map(normalize_team)
    odds["away_team"] = odds["Away Team"].map(normalize_team)

    # Keep relevant odds columns
    odds_cols = {
        "Home Odds": "odds_home",
        "Away Odds": "odds_away",
        "Home Odds Close": "odds_home_close",
        "Away Odds Close": "odds_away_close",
        "Play Off Game?": "is_final",
    }
    for old, new in odds_cols.items():
        if old in odds.columns:
            odds[new] = odds[old]
    
    if "is_final" in odds.columns:
        odds["is_final"] = (
            odds["is_final"].replace({"Y": 1}).fillna(0).astype(int)
        )
    else:
        odds["is_final"] = 0
    
    keep = ["date", "home_team", "away_team"] + list(odds_cols.values())
    odds = odds[[c for c in keep if c in odds.columns]].copy()
    return odds
        

def merge_data(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge match results with odds on (date, home_team, away_team)."""
    print(f"  Matches: {len(matches)}, Odds: {len(odds)}")
    merged = matches.merge(
        odds,
        on=["date", "home_team", "away_team"],
        how="inner",
    )
    print(f"  Merged: {len(merged)}")
    if len(merged) == 0:
        print("  WARNING: Merge resulted in 0 rows!")
        print(f"  Matches teams: {matches['home_team'].unique()[:5]}")
        print(f"  Odds teams: {odds['home_team'].unique()[:5]}")
        return merged

    # Convert odds to implied probabilities
    merged["implied_prob_home"] = 1.0 / merged["odds_home"]
    merged["implied_prob_away"] = 1.0 / merged["odds_away"]
    # Normalize to remove overround
    total = merged["implied_prob_home"] + merged["implied_prob_away"]
    merged["market_prob_home"] = merged["implied_prob_home"] / total
    merged["market_prob_away"] = merged["implied_prob_away"] / total

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def run():
    """Main ingestion pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading match data...")
    matches = download_match_data()
    print(f"  {len(matches)} matches downloaded")

    print("Downloading odds data...")
    odds = download_odds_data()
    print(f"  {len(odds)} odds records downloaded")

    print("Merging...")
    merged = merge_data(matches, odds)
    print(f"  {len(merged)} merged records")
    print(f"  Year range: {merged['year'].min()}-{merged['year'].max()}")
    print(f"  Teams: {sorted(merged['home_team'].unique())}")

    merged.to_parquet(MERGED_PATH, index=False)
    print(f"Saved to {MERGED_PATH}")
    return merged


if __name__ == "__main__":
    run()
