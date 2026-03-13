"""Feature engineering: Elo ratings, rolling stats, venue/rest features."""

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_ADV,
    ELO_INIT,
    ELO_K,
    ELO_SEASON_REVERT,
    FEATURE_COLS,
    FEATURE_PATH,
    MERGED_PATH,
)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for side A."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _match_result_from_margin(margin: float) -> float:
    """Binary-style result with draw support."""
    if margin > 0:
        return 1.0
    if margin < 0:
        return 0.0
    return 0.5


def _prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Sort matches deterministically before any sequential feature building."""
    sort_cols = ["date", "year", "home_team", "away_team"]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _team_long_history(df: pd.DataFrame) -> pd.DataFrame:
    """Create a team-centric history table with one row per team per match."""
    home_result = df["margin"].apply(_match_result_from_margin)
    away_result = 1.0 - home_result

    home_rows = df[
        [
            "date",
            "year",
            "round_num",
            "venue",
            "home_team",
            "away_team",
            "margin",
            "home_score",
            "elo_home",
            "elo_home_post",
        ]
    ].copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["win"] = home_result.values
    home_rows["team_margin"] = home_rows["margin"]
    home_rows["team_score"] = home_rows["home_score"]
    home_rows["elo_pre"] = home_rows["elo_home"]
    home_rows["elo_post"] = home_rows["elo_home_post"]
    home_rows["is_home"] = 1
    home_rows["match_idx"] = home_rows.index

    away_rows = df[
        [
            "date",
            "year",
            "round_num",
            "venue",
            "home_team",
            "away_team",
            "margin",
            "away_score",
            "elo_away",
            "elo_away_post",
        ]
    ].copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["win"] = away_result.values
    away_rows["team_margin"] = -away_rows["margin"]
    away_rows["team_score"] = away_rows["away_score"]
    away_rows["elo_pre"] = away_rows["elo_away"]
    away_rows["elo_post"] = away_rows["elo_away_post"]
    away_rows["is_home"] = 0
    away_rows["match_idx"] = away_rows.index

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    return long.sort_values(["team", "date", "match_idx"], kind="mergesort").reset_index(
        drop=True
    )


def _build_h2h_feature(df: pd.DataFrame) -> pd.Series:
    """Home-team historical win rate versus the away team, regardless of venue."""
    records = {}
    values = []

    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        pair = tuple(sorted((home, away)))
        home_wins, total = records.get(pair, (0.0, 0))
        if total == 0:
            values.append(0.5)
        else:
            if home == pair[0]:
                values.append(home_wins / total)
            else:
                values.append(1.0 - (home_wins / total))

        result = _match_result_from_margin(row.margin)
        if home == pair[0]:
            records[pair] = (home_wins + result, total + 1)
        else:
            records[pair] = (home_wins + (1.0 - result), total + 1)

    return pd.Series(values, index=df.index, dtype=float)


def build_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-match and post-match Elo ratings for each match."""
    df = _prepare_matches(df).copy()
    ratings = {}
    elo_home_list = []
    elo_away_list = []
    elo_home_post = []
    elo_away_post = []
    prev_season = None

    for row in df.itertuples(index=False):
        season = row.year
        home = row.home_team
        away = row.away_team

        if prev_season is not None and season != prev_season:
            for team in list(ratings.keys()):
                ratings[team] = (
                    ratings[team] * (1 - ELO_SEASON_REVERT)
                    + ELO_INIT * ELO_SEASON_REVERT
                )
        prev_season = season

        r_home = ratings.get(home, ELO_INIT)
        r_away = ratings.get(away, ELO_INIT)
        elo_home_list.append(r_home)
        elo_away_list.append(r_away)

        exp_home = _elo_expected(r_home + ELO_HOME_ADV, r_away)
        actual_home = _match_result_from_margin(row.margin)
        new_home = r_home + ELO_K * (actual_home - exp_home)
        new_away = r_away + ELO_K * ((1 - actual_home) - (1 - exp_home))

        ratings[home] = new_home
        ratings[away] = new_away
        elo_home_post.append(new_home)
        elo_away_post.append(new_away)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    df["elo_home_post"] = elo_home_post
    df["elo_away_post"] = elo_away_post
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    df["elo_prob"] = _elo_expected(
        df["elo_home"] + ELO_HOME_ADV,
        df["elo_away"],
    )
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rolling, matchup, and derived features without future leakage."""
    df = _prepare_matches(df).copy()
    long = _team_long_history(df)

    grouped = long.groupby("team", sort=False)
    long["form_5"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    long["win_pct_10"] = grouped["win"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    long["margin_ewma"] = grouped["team_margin"].transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    long["scoring_ewma"] = grouped["team_score"].transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False, min_periods=1).mean()
    )
    long["venue_exp"] = grouped["venue"].transform(
        lambda s: s.groupby(s).cumcount().astype(float)
    )
    long["rest_days"] = grouped["date"].transform(
        lambda s: s.diff().dt.days.clip(lower=0, upper=30)
    )

    df["h2h_home_win_pct"] = _build_h2h_feature(df)

    home_feats = long[long["is_home"] == 1].set_index("match_idx")[
        ["form_5", "win_pct_10", "margin_ewma", "scoring_ewma", "venue_exp", "rest_days"]
    ]
    home_feats.columns = [
        "form_home_5",
        "win_pct_home_10",
        "margin_ewma_home",
        "scoring_ewma_home",
        "venue_exp_home",
        "rest_days_home",
    ]

    away_feats = long[long["is_home"] == 0].set_index("match_idx")[
        ["form_5", "win_pct_10", "margin_ewma", "scoring_ewma", "venue_exp", "rest_days"]
    ]
    away_feats.columns = [
        "form_away_5",
        "win_pct_away_10",
        "margin_ewma_away",
        "scoring_ewma_away",
        "venue_exp_away",
        "rest_days_away",
    ]

    df = df.join(home_feats).join(away_feats)
    df["form_diff"] = df["form_home_5"] - df["form_away_5"]
    df["win_pct_diff"] = df["win_pct_home_10"] - df["win_pct_away_10"]
    df["venue_exp_diff"] = df["venue_exp_home"] - df["venue_exp_away"]
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["margin_ewma_diff"] = df["margin_ewma_home"] - df["margin_ewma_away"]
    df["scoring_ewma_diff"] = df["scoring_ewma_home"] - df["scoring_ewma_away"]
    df["market_elo_delta"] = df["market_prob_home"] - df["elo_prob"]

    finals_map = {
        "Elimination Final": 25,
        "Qualifying Final": 25,
        "Semi Final": 26,
        "Preliminary Final": 27,
        "Grand Final": 28,
    }
    df["season_round"] = df["round_num"].map(
        lambda x: finals_map.get(x, int(x) if str(x).isdigit() else 25)
    )

    fill_values = {
        "form_home_5": 0.5,
        "form_away_5": 0.5,
        "form_diff": 0.0,
        "win_pct_home_10": 0.5,
        "win_pct_away_10": 0.5,
        "win_pct_diff": 0.0,
        "venue_exp_home": 0.0,
        "venue_exp_away": 0.0,
        "venue_exp_diff": 0.0,
        "rest_days_home": 30.0,
        "rest_days_away": 30.0,
        "rest_diff": 0.0,
        "h2h_home_win_pct": 0.5,
        "margin_ewma_home": 0.0,
        "margin_ewma_away": 0.0,
        "margin_ewma_diff": 0.0,
        "scoring_ewma_home": 80.0,
        "scoring_ewma_away": 80.0,
        "scoring_ewma_diff": 0.0,
        "market_elo_delta": 0.0,
    }
    for column, value in fill_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    return df


def _team_history_for_snapshot(df: pd.DataFrame, team: str, as_of_date=None) -> pd.DataFrame:
    """Return prior matches for a team from either home or away perspective."""
    home = df[df["home_team"] == team].copy()
    if not home.empty:
        home["team"] = team
        home["opponent"] = home["away_team"]
        home["win"] = home["margin"].apply(_match_result_from_margin)
        home["team_margin"] = home["margin"]
        home["team_score"] = home["home_score"]
        home["elo_pre"] = home["elo_home"]
        home["elo_post"] = home["elo_home_post"]

    away = df[df["away_team"] == team].copy()
    if not away.empty:
        away["team"] = team
        away["opponent"] = away["home_team"]
        away["win"] = away["margin"].apply(lambda m: 1.0 - _match_result_from_margin(m))
        away["team_margin"] = -away["margin"]
        away["team_score"] = away["away_score"]
        away["elo_pre"] = away["elo_away"]
        away["elo_post"] = away["elo_away_post"]

    history = pd.concat([home, away], ignore_index=True, sort=False)
    if history.empty:
        return history

    history = history.sort_values(["date", "year", "home_team", "away_team"], kind="mergesort")
    if as_of_date is not None:
        history = history[history["date"] < pd.Timestamp(as_of_date)]
    return history.reset_index(drop=True)


def _team_snapshot(
    df: pd.DataFrame,
    team: str,
    venue: str = None,
    match_date=None,
) -> dict:
    """Build the current pre-match state for a single team."""
    history = _team_history_for_snapshot(df, team, as_of_date=match_date)
    if history.empty:
        return {
            "elo": ELO_INIT,
            "form_5": 0.5,
            "win_pct_10": 0.5,
            "margin_ewma": 0.0,
            "scoring_ewma": 80.0,
            "venue_exp": 0.0,
            "rest_days": 30.0,
        }

    wins = history["win"]
    margins = history["team_margin"]
    scores = history["team_score"]
    last_date = history["date"].iloc[-1]
    if match_date is None:
        rest_days = 7.0
    else:
        rest_days = float(np.clip((pd.Timestamp(match_date) - last_date).days, 0, 30))

    venue_exp = 0.0
    if venue:
        venue_exp = float((history["venue"] == venue).sum())

    return {
        "elo": float(history["elo_post"].iloc[-1]),
        "form_5": float(wins.tail(5).mean()),
        "win_pct_10": float(wins.tail(10).mean()),
        "margin_ewma": float(margins.ewm(span=10, adjust=False).mean().iloc[-1]),
        "scoring_ewma": float(scores.ewm(span=10, adjust=False).mean().iloc[-1]),
        "venue_exp": venue_exp,
        "rest_days": rest_days,
    }


def _current_h2h_prob(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    match_date=None,
) -> float:
    """Home-team win rate versus the away team before the given match date."""
    history = df[
        (
            ((df["home_team"] == home_team) & (df["away_team"] == away_team))
            | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
        )
    ].copy()
    if match_date is not None:
        history = history[history["date"] < pd.Timestamp(match_date)]
    if history.empty:
        return 0.5

    outcomes = []
    for row in history.itertuples(index=False):
        result = _match_result_from_margin(row.margin)
        outcomes.append(result if row.home_team == home_team else 1.0 - result)
    return float(np.mean(outcomes))


def build_current_match_features(
    history_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    odds_home: float = None,
    odds_away: float = None,
    venue: str = None,
    match_date=None,
    season_round: int = 1,
) -> dict | None:
    """Build a feature dict for a future match from historical data only."""
    if history_df.empty:
        return None

    home = _team_snapshot(history_df, home_team, venue=venue, match_date=match_date)
    away = _team_snapshot(history_df, away_team, venue=venue, match_date=match_date)
    elo_prob = _elo_expected(home["elo"] + ELO_HOME_ADV, away["elo"])

    if odds_home and odds_away and odds_home > 0 and odds_away > 0:
        implied_home = 1.0 / odds_home
        implied_away = 1.0 / odds_away
        market_overround = implied_home + implied_away
        market_prob_home = implied_home / market_overround
        market_prob_away = implied_away / market_overround
    else:
        market_prob_home = 0.5
        market_prob_away = 0.5
        market_overround = 1.0

    features = {
        "elo_diff": home["elo"] - away["elo"],
        "elo_prob": elo_prob,
        "form_home_5": home["form_5"],
        "form_away_5": away["form_5"],
        "form_diff": home["form_5"] - away["form_5"],
        "win_pct_home_10": home["win_pct_10"],
        "win_pct_away_10": away["win_pct_10"],
        "win_pct_diff": home["win_pct_10"] - away["win_pct_10"],
        "venue_exp_home": home["venue_exp"],
        "venue_exp_away": away["venue_exp"],
        "venue_exp_diff": home["venue_exp"] - away["venue_exp"],
        "rest_days_home": home["rest_days"],
        "rest_days_away": away["rest_days"],
        "rest_diff": home["rest_days"] - away["rest_days"],
        "h2h_home_win_pct": _current_h2h_prob(
            history_df, home_team, away_team, match_date=match_date
        ),
        "season_round": season_round,
        "margin_ewma_home": home["margin_ewma"],
        "margin_ewma_away": away["margin_ewma"],
        "margin_ewma_diff": home["margin_ewma"] - away["margin_ewma"],
        "scoring_ewma_home": home["scoring_ewma"],
        "scoring_ewma_away": away["scoring_ewma"],
        "scoring_ewma_diff": home["scoring_ewma"] - away["scoring_ewma"],
        "market_prob_home": market_prob_home,
        "market_prob_away": market_prob_away,
        "market_overround": market_overround,
        "market_elo_delta": market_prob_home - elo_prob,
    }
    return features


def build_feature_matrix(
    merged_path: str = MERGED_PATH,
    output_path: str = FEATURE_PATH,
) -> pd.DataFrame:
    """Full pipeline: load merged data, compute all features, save."""
    print("Loading merged data...")
    df = pd.read_parquet(merged_path)
    df = _prepare_matches(df)

    print("Building Elo ratings...")
    df = build_elo(df)

    print("Building rolling features...")
    df = build_rolling_features(df)

    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    before = len(df)
    df = df.dropna(subset=["home_win", "odds_home", "odds_away"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with missing targets or odds")

    print(f"Feature matrix: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if output_path:
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    build_feature_matrix()
