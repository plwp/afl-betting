"""Feature engineering: Elo ratings, rolling stats, venue/rest features."""

import numpy as np
import pandas as pd
from config import (
    ELO_K, ELO_HOME_ADV, ELO_INIT, ELO_SEASON_REVERT,
    FEATURE_PATH, MERGED_PATH, FEATURE_COLS,
)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for player A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def build_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for each match.

    Ratings are stored *before* the match result is applied (i.e., the
    rating used for prediction, not the post-match rating).
    """
    ratings = {}
    elo_home_list = []
    elo_away_list = []
    prev_season = None

    for _, row in df.iterrows():
        season = row["year"]
        home, away = row["home_team"], row["away_team"]

        # Season mean reversion
        if prev_season is not None and season != prev_season:
            for team in list(ratings.keys()):
                ratings[team] = (
                    ratings[team] * (1 - ELO_SEASON_REVERT)
                    + ELO_INIT * ELO_SEASON_REVERT
                )
        prev_season = season

        r_home = ratings.get(home, ELO_INIT)
        r_away = ratings.get(away, ELO_INIT)

        # Record pre-match ratings
        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        # Expected with home advantage
        exp_home = _elo_expected(r_home + ELO_HOME_ADV, r_away)
        actual_home = 1.0 if row["margin"] > 0 else (0.5 if row["margin"] == 0 else 0.0)

        # Margin-based K multiplier
        margin = abs(row["margin"])
        # For AFL, a winning margin of ~30-40 is 'typical'.
        # We use a log-scaled margin to prevent blowouts from overly biasing Elo.
        k_multiplier = np.log1p(margin) * (2.2 / ((r_home - r_away) * 0.001 + 2.2))
        k = ELO_K * k_multiplier

        # Update
        ratings[home] = r_home + k * (actual_home - exp_home)
        ratings[away] = r_away + k * ((1 - actual_home) - (1 - exp_home))

    df = df.copy()
    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_prob"] = df.apply(
        lambda r: _elo_expected(r["elo_home"] + ELO_HOME_ADV, r["elo_away"]),
        axis=1,
    )
    # New: diff from market
    df["elo_market_diff"] = df["elo_prob"] - df["market_prob_home"]
    return df


def _rolling_team_stat(df: pd.DataFrame, team_col: str, stat_col: str,
                       window: int, team: str) -> pd.Series:
    """Compute rolling mean for a team, shifted by 1 to prevent leakage."""
    mask = df[team_col] == team
    return df.loc[mask, stat_col].shift(1).rolling(window, min_periods=1).mean()


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all rolling and derived features."""
    df = df.copy()
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))

    # --- Per-team rolling stats ---
    # We need to build team-level time series, then map back to matches.
    # Create a long-form dataframe: one row per team per match.
    home_rows = df[["date", "year", "round_num", "home_team", "away_team",
                    "home_win", "margin", "home_score", "venue"]].copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["win"] = home_rows["home_win"]
    home_rows["team_margin"] = home_rows["margin"]
    home_rows["team_score"] = home_rows["home_score"]
    home_rows["is_home"] = 1
    home_rows["match_idx"] = home_rows.index

    away_rows = df[["date", "year", "round_num", "home_team", "away_team",
                    "home_win", "margin", "away_score", "venue"]].copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["win"] = 1 - away_rows["home_win"]
    away_rows["team_margin"] = -away_rows["margin"]
    away_rows["team_score"] = away_rows["away_score"]
    away_rows["is_home"] = 0
    away_rows["match_idx"] = away_rows.index

    long = pd.concat([home_rows, away_rows]).sort_values(["team", "date"]).reset_index(drop=True)

    # Shifted rolling features per team (shift(1) prevents leakage)
    for team in teams:
        mask = long["team"] == team
        idx = long.index[mask]

        # Form: rolling win rate over last 5
        long.loc[idx, "form_5"] = (
            long.loc[idx, "win"].shift(1).rolling(5, min_periods=1).mean()
        )
        # Win pct over last 10
        long.loc[idx, "win_pct_10"] = (
            long.loc[idx, "win"].shift(1).rolling(10, min_periods=1).mean()
        )
        # Margin EWMA
        long.loc[idx, "margin_ewma"] = (
            long.loc[idx, "team_margin"].shift(1).ewm(span=10, min_periods=1).mean()
        )
        # Scoring EWMA
        long.loc[idx, "scoring_ewma"] = (
            long.loc[idx, "team_score"].shift(1).ewm(span=10, min_periods=1).mean()
        )
        # Venue experience: cumulative games at this venue
        venue_counts = long.loc[idx].groupby("venue").cumcount()
        long.loc[idx, "venue_exp"] = venue_counts.values

        # Rest days
        long.loc[idx, "rest_days"] = (
            long.loc[idx, "date"].diff().dt.days.clip(upper=30)
        )

    # --- H2H ---
    # For each match, compute home team's historical win rate vs away team (any venue)
    h2h_records = {}
    h2h_col = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        # Use a sorted tuple to track H2H regardless of who is home
        teams_key = tuple(sorted([h, a]))
        wins, total = h2h_records.get(teams_key, (0, 0))
        
        # We want the win rate of the home_team against the away_team
        # If total is 0, use 0.5
        if total > 0:
            # We need to know which team in the key is the 'home_team'
            # Let's store wins as (count of team1 wins, count of team2 wins)
            # but that's complex. Let's just store a simple dict of dicts.
            pass
            
    # Redoing H2H more cleanly:
    h2h_stats = {} # (teamA, teamB) -> [wins_A, total]
    h2h_col = []
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        
        # Get stats for h vs a
        stats = h2h_stats.get((h, a), [0, 0])
        rev_stats = h2h_stats.get((a, h), [0, 0])
        
        total_vs = stats[1] + rev_stats[1]
        wins_vs = stats[0] + (rev_stats[1] - rev_stats[0]) # h's wins = (h as home wins) + (a as home losses)
        
        h2h_col.append(wins_vs / total_vs if total_vs > 0 else 0.5)
        
        # Update records
        res = 1 if row["home_win"] == 1 else (0.5 if row["margin"] == 0 else 0)
        curr = h2h_stats.get((h, a), [0, 0])
        h2h_stats[(h, a)] = [curr[0] + res, curr[1] + 1]

    df["h2h_home_win_pct"] = h2h_col

    # --- Map long-form features back to match rows ---
    home_feats = long[long["is_home"] == 1].set_index("match_idx")[
        ["form_5", "win_pct_10", "margin_ewma", "scoring_ewma", "venue_exp", "rest_days"]
    ].rename(columns=lambda c: c + "_home" if not c.endswith("_home") else c)

    # Fix column names
    home_feats.columns = [
        "form_home_5", "win_pct_home_10", "margin_ewma_home",
        "scoring_ewma_home", "venue_exp_home", "rest_days_home",
    ]

    away_feats = long[long["is_home"] == 0].set_index("match_idx")[
        ["form_5", "win_pct_10", "margin_ewma", "scoring_ewma", "venue_exp", "rest_days"]
    ]
    away_feats.columns = [
        "form_away_5", "win_pct_away_10", "margin_ewma_away",
        "scoring_ewma_away", "venue_exp_away", "rest_days_away",
    ]

    df = df.join(home_feats).join(away_feats)

    # Derived
    df["form_diff"] = df["form_home_5"] - df["form_away_5"]
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]

    # --- Travel and State ---
    from config import TEAM_STATE, VENUE_STATE
    
    # State-to-state flight time approximation (hours)
    # Simple lookup for dist between team state and venue state
    # (Default 1.0)
    FLIGHT_TIMES = {
        ("VIC", "SA"): 1.0, ("VIC", "NSW"): 1.0, ("VIC", "TAS"): 1.0, 
        ("VIC", "QLD"): 2.0, ("VIC", "WA"): 4.0, ("VIC", "NT"): 3.0,
        ("SA", "VIC"): 1.0, ("SA", "NSW"): 1.5, ("SA", "QLD"): 2.5, ("SA", "WA"): 3.0,
        ("WA", "VIC"): 4.0, ("WA", "SA"): 3.0, ("WA", "NSW"): 4.5, ("WA", "QLD"): 5.0,
        ("NSW", "VIC"): 1.0, ("NSW", "SA"): 1.5, ("NSW", "QLD"): 1.5, ("NSW", "WA"): 4.5,
        ("QLD", "VIC"): 2.0, ("QLD", "SA"): 2.5, ("QLD", "NSW"): 1.5, ("QLD", "WA"): 5.0,
    }

    def get_travel_dist(team, venue):
        ts = TEAM_STATE.get(team)
        vs = VENUE_STATE.get(venue)
        if not ts or not vs: return 1.0
        if ts == vs: return 0.0
        return FLIGHT_TIMES.get((ts, vs), 1.5)

    df["is_home_state_home"] = df.apply(
        lambda r: 1 if TEAM_STATE.get(r["home_team"]) == VENUE_STATE.get(r["venue"]) else 0,
        axis=1
    )
    df["is_home_state_away"] = df.apply(
        lambda r: 1 if TEAM_STATE.get(r["away_team"]) == VENUE_STATE.get(r["venue"]) else 0,
        axis=1
    )
    df["travel_dist_home"] = df.apply(lambda r: get_travel_dist(r["home_team"], r["venue"]), axis=1)
    df["travel_dist_away"] = df.apply(lambda r: get_travel_dist(r["away_team"], r["venue"]), axis=1)

    # Convert round_num to numeric
    finals_map = {
        "Elimination Final": 25,
        "Qualifying Final": 25,
        "Semi Final": 26,
        "Preliminary Final": 27,
        "Grand Final": 28,
    }
    def clean_round(x):
        if str(x).isdigit(): return int(x)
        # Handle "Round X"
        if "Round" in str(x):
            try: return int(str(x).split()[-1])
            except: pass
        return finals_map.get(x, 25)

    df["season_round"] = df["round_num"].map(clean_round)

    return df


def build_feature_matrix(merged_path: str = MERGED_PATH,
                         output_path: str = FEATURE_PATH) -> pd.DataFrame:
    """Full pipeline: load merged data, compute all features, save."""
    print("Loading merged data...")
    df = pd.read_parquet(merged_path)

    print("Building Elo ratings...")
    df = build_elo(df)

    print("Building rolling features...")
    df = build_rolling_features(df)

    # Drop rows with NaN in feature columns (early games with insufficient history)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with missing features")

    print(f"Feature matrix: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    build_feature_matrix()
