"""Download and merge AFL match data with historical odds."""

import os
import pandas as pd
import requests
from io import BytesIO, StringIO
from config import (
    MATCH_CSV_URL, ODDS_XLSX_URL, MATCH_YEARS,
    DATA_DIR, MERGED_PATH, TEAM_NAME_MAP,
)


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def download_match_data() -> pd.DataFrame:
    frames = []
    for year in MATCH_YEARS:
        url = MATCH_CSV_URL.format(year=year)
        print(f"  Downloading matches {year}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        frames.append(pd.read_csv(StringIO(resp.text)))

    matches = pd.concat(frames, ignore_index=True)
    matches["home_score"] = matches["team_1_final_goals"] * 6 + matches["team_1_final_behinds"]
    matches["away_score"] = matches["team_2_final_goals"] * 6 + matches["team_2_final_behinds"]
    matches["margin"] = matches["home_score"] - matches["away_score"]
    matches["home_win"] = (matches["margin"] > 0).astype(int)
    matches["home_team"] = matches["team_1_team_name"].map(normalize_team)
    matches["away_team"] = matches["team_2_team_name"].map(normalize_team)
    matches["date"] = pd.to_datetime(matches["date"].str.strip(), format="mixed").dt.normalize()
    matches["year"] = matches["year"].astype(int)
    matches["round_num"] = matches["round_num"].astype(str)

    keep = [
        "date", "year", "round_num", "venue",
        "home_team", "away_team",
        "home_score", "away_score", "margin", "home_win",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]
    return matches[keep].copy()


def download_odds_data() -> pd.DataFrame:
    print("  Downloading odds data...")
    resp = requests.get(ODDS_XLSX_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    odds = pd.read_excel(BytesIO(resp.content), header=1)

    odds["date"] = pd.to_datetime(odds["Date"]).dt.normalize()
    odds["home_team"] = odds["Home Team"].map(normalize_team)
    odds["away_team"] = odds["Away Team"].map(normalize_team)

    odds_cols = {
        "Home Odds": "odds_home",
        "Away Odds": "odds_away",
        "Home Odds Close": "odds_home_close",
        "Away Odds Close": "odds_away_close",
        "Home Line Open": "home_line_open",
        "Home Line Close": "home_line_close",
        "Home Line Odds Close": "home_line_odds_close",
        "Away Line Odds Close": "away_line_odds_close",
        "Play Off Game?": "is_final",
    }
    for old, new in odds_cols.items():
        if old in odds.columns:
            odds[new] = odds[old]

    if "is_final" in odds.columns:
        odds["is_final"] = odds["is_final"].replace({"Y": 1}).fillna(0).astype(int)
    else:
        odds["is_final"] = 0

    keep = ["date", "home_team", "away_team"] + list(odds_cols.values())
    return odds[[c for c in keep if c in odds.columns]].copy()


def merge_data(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = matches.merge(odds, on=["date", "home_team", "away_team"], how="inner")

    merged["implied_prob_home"] = 1.0 / merged["odds_home"]
    merged["implied_prob_away"] = 1.0 / merged["odds_away"]
    total = merged["implied_prob_home"] + merged["implied_prob_away"]
    merged["market_prob_home"] = merged["implied_prob_home"] / total
    merged["market_prob_away"] = merged["implied_prob_away"] / total
    merged["market_overround"] = total

    return merged.sort_values("date").reset_index(drop=True)


def run():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading match data...")
    matches = download_match_data()
    print(f"  {len(matches)} matches downloaded")
    print("Downloading odds data...")
    odds = download_odds_data()
    print(f"  {len(odds)} odds records downloaded")
    print("Merging...")
    merged = merge_data(matches, odds)
    print(f"  {len(merged)} merged records ({merged['year'].min()}-{merged['year'].max()})")
    merged.to_parquet(MERGED_PATH, index=False)
    print(f"Saved to {MERGED_PATH}")
    return merged


if __name__ == "__main__":
    run()
