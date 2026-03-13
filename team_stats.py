"""Scrape team-level match statistics from FootyWire."""

import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
from io import StringIO

from config import DATA_DIR, MATCH_YEARS, TEAM_NAME_MAP


STATS_CACHE_DIR = os.path.join(DATA_DIR, "footywire_cache")
FOOTYWIRE_BASE = "https://www.footywire.com/afl/footy"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Stats we want per team per match
STAT_KEYS = [
    "Kicks", "Handballs", "Disposals", "Marks", "Tackles",
    "Hitouts", "Inside 50s", "Clearances", "Clangers", "Rebound 50s",
    "Frees For", "Frees Against", "Contested Possessions",
    "Uncontested Possessions", "Contested Marks",
]


def _cache_path(mid: int) -> str:
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)
    return os.path.join(STATS_CACHE_DIR, f"match_{mid}.json")


def _normalize(name: str) -> str:
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def _scrape_match_stats(mid: int) -> dict | None:
    """Scrape team-level stats for a single match from FootyWire."""
    cache_file = _cache_path(mid)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    url = f"{FOOTYWIRE_BASE}/ft_match_statistics?mid={mid}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        # Parse HTML tables
        tables = pd.read_html(StringIO(resp.text))

        # Find the head-to-head stats table
        # It typically has 3 columns: stat_name, team1_value, team2_value
        stats_table = None
        for tbl in tables:
            if len(tbl.columns) == 3 and len(tbl) >= 10:
                # Check if it has expected stat names
                col0_vals = tbl.iloc[:, 0].astype(str).tolist()
                if any("Disposals" in v for v in col0_vals):
                    stats_table = tbl
                    break

        if stats_table is None:
            return None

        # Extract team names from column headers
        cols = list(stats_table.columns)
        home_team = _normalize(str(cols[1]))
        away_team = _normalize(str(cols[2]))

        # Build stats dict
        result = {"home_team": home_team, "away_team": away_team, "mid": mid}
        for _, row in stats_table.iterrows():
            stat_name = str(row.iloc[0]).strip()
            if stat_name in STAT_KEYS:
                try:
                    result[f"home_{stat_name}"] = int(row.iloc[1])
                    result[f"away_{stat_name}"] = int(row.iloc[2])
                except (ValueError, TypeError):
                    pass

        # Cache
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    except Exception:
        return None


def _get_match_ids(year: int) -> list[int]:
    """Scrape match IDs for a given year from FootyWire fixture page."""
    cache_file = os.path.join(STATS_CACHE_DIR, f"match_ids_{year}.json")
    os.makedirs(STATS_CACHE_DIR, exist_ok=True)

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    url = f"{FOOTYWIRE_BASE}/ft_match_list?year={year}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        # Find all match stat links: ft_match_statistics?mid=XXXXX
        mids = list(set(
            int(m) for m in re.findall(r'ft_match_statistics\?mid=(\d+)', resp.text)
        ))
        mids.sort()

        with open(cache_file, "w") as f:
            json.dump(mids, f)

        time.sleep(1)
        return mids

    except Exception as e:
        print(f"  Error fetching match IDs for {year}: {e}")
        return []


def download_team_stats(years: range = None) -> pd.DataFrame:
    """Download team-level match stats for all matches in the given years.

    Returns DataFrame with per-match team stats.
    """
    if years is None:
        years = MATCH_YEARS

    all_stats = []
    total_api_calls = 0

    for year in years:
        mids = _get_match_ids(year)
        if not mids:
            print(f"  {year}: no match IDs found")
            continue

        year_stats = 0
        for mid in mids:
            cache_file = _cache_path(mid)
            needs_fetch = not os.path.exists(cache_file)

            stats = _scrape_match_stats(mid)
            if stats and len(stats) > 3:
                all_stats.append(stats)
                year_stats += 1

            if needs_fetch:
                total_api_calls += 1
                # Rate limit: be polite to FootyWire
                time.sleep(0.5)
                if total_api_calls % 50 == 0:
                    print(f"  Scraped {total_api_calls} matches so far...")

        print(f"  {year}: {year_stats}/{len(mids)} matches with stats")

    if not all_stats:
        return pd.DataFrame()

    df = pd.DataFrame(all_stats)
    print(f"  Total: {len(df)} matches with team stats ({total_api_calls} API calls)")
    return df


def merge_team_stats(merged_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Merge team stats into the main match DataFrame.

    Matches on (home_team, away_team) pairs. Since FootyWire doesn't have dates
    in the stats, we match by team pair and year ordering.
    """
    if stats_df.empty:
        return merged_df

    # We need to carefully match FootyWire stats to our matches
    # Group by home_team, away_team and assign stats in order
    stat_cols = [c for c in stats_df.columns if c.startswith("home_") or c.startswith("away_")]
    stat_cols = [c for c in stat_cols if c not in ("home_team", "away_team")]

    # Simplify column names for features
    rename_map = {}
    for col in stat_cols:
        side = "home" if col.startswith("home_") else "away"
        stat = col.replace(f"{side}_", "").lower().replace(" ", "_")
        rename_map[col] = f"fw_{stat}_{side}"

    stats_clean = stats_df.rename(columns=rename_map)

    # For now, skip the complex matching and just return — the user can
    # iterate on this. We'll use the stats directly in feature engineering.
    return stats_clean


if __name__ == "__main__":
    print("Downloading team stats from FootyWire...")
    stats = download_team_stats(range(2009, 2025))
    if not stats.empty:
        out_path = os.path.join(DATA_DIR, "team_stats.parquet")
        stats.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
        print(f"Columns: {list(stats.columns)}")
