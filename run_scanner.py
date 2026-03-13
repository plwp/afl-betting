#!/usr/bin/env python3
"""Entry point: scan live odds for value bets."""

import argparse
import os
import joblib
import pandas as pd
from tabulate import tabulate

from config import (
    FEATURE_PATH, MODEL_DIR, FEATURE_COLS,
    INITIAL_BANKROLL, EDGE_THRESHOLD,
)
from features import build_elo, build_rolling_features
from scanner import fetch_odds, parse_odds, scan_value_bets
from tracker import BetTracker


def build_current_features(feature_df: pd.DataFrame,
                           home_team: str, away_team: str) -> dict:
    """Extract latest features for a given matchup from historical data.

    Uses the most recent feature values for each team.
    """
    # Get last row where this team played at home / away
    home_rows = feature_df[feature_df["home_team"] == home_team]
    away_rows = feature_df[feature_df["away_team"] == away_team]

    if home_rows.empty or away_rows.empty:
        return None

    last_home = home_rows.iloc[-1]
    last_away = away_rows.iloc[-1]

    # Build feature dict from latest data
    features = {
        "elo_diff": last_home.get("elo_home", 1500) - last_away.get("elo_away", 1500),
        "elo_prob": last_home.get("elo_prob", 0.5),
        "form_home_5": last_home.get("form_home_5", 0.5),
        "form_away_5": last_away.get("form_away_5", 0.5),
        "win_pct_home_10": last_home.get("win_pct_home_10", 0.5),
        "win_pct_away_10": last_away.get("win_pct_away_10", 0.5),
        "venue_exp_home": last_home.get("venue_exp_home", 0),
        "venue_exp_away": last_away.get("venue_exp_away", 0),
        "rest_days_home": last_home.get("rest_days_home", 7),
        "rest_days_away": last_away.get("rest_days_away", 7),
        "h2h_home_win_pct": last_home.get("h2h_home_win_pct", 0.5),
        "season_round": 1,  # default for upcoming
        "margin_ewma_home": last_home.get("margin_ewma_home", 0),
        "margin_ewma_away": last_away.get("margin_ewma_away", 0),
        "scoring_ewma_home": last_home.get("scoring_ewma_home", 80),
        "scoring_ewma_away": last_away.get("scoring_ewma_away", 80),
    }
    features["form_diff"] = features["form_home_5"] - features["form_away_5"]
    features["rest_diff"] = features["rest_days_home"] - features["rest_days_away"]
    return features


def main():
    parser = argparse.ArgumentParser(description="AFL Value Bet Scanner")
    parser.add_argument("--edge", type=float, default=EDGE_THRESHOLD,
                        help="Minimum edge threshold")
    parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL,
                        help="Current bankroll")
    parser.add_argument("--log", action="store_true",
                        help="Log bets to tracker database")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh odds (ignore cache)")
    args = parser.parse_args()

    # Load model
    model_path = os.path.join(MODEL_DIR, "lgb_calibrated.pkl")
    if not os.path.exists(model_path):
        print("No trained model found. Run model.py or run_backtest.py first.")
        return

    model = joblib.load(model_path)

    # Load feature matrix for current team stats
    if not os.path.exists(FEATURE_PATH):
        print("No feature matrix found. Run features.py first.")
        return
    feature_df = pd.read_parquet(FEATURE_PATH)

    # Fetch odds
    try:
        events = fetch_odds(force_refresh=args.refresh)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"API error: {e}")
        return

    if not events:
        print("No upcoming AFL events found.")
        return

    odds_df = parse_odds(events)
    print(f"\n{len(odds_df)} upcoming matches found")

    # Build model predictions for each matchup
    model_probs = {}
    for _, row in odds_df.iterrows():
        feats = build_current_features(feature_df, row["home_team"], row["away_team"])
        if feats is None:
            continue
        X = pd.DataFrame([feats])[FEATURE_COLS].values
        prob = model.predict_proba(X)[0, 1]
        model_probs[(row["home_team"], row["away_team"])] = prob

    # Scan for value
    value_bets = scan_value_bets(odds_df, model_probs, args.bankroll, args.edge)

    if value_bets.empty:
        print("\nNo value bets found above edge threshold.")
        # Still show all matches with model probs
        print("\nAll matches:")
        for _, row in odds_df.iterrows():
            key = (row["home_team"], row["away_team"])
            prob = model_probs.get(key, None)
            if prob:
                print(f"  {row['home_team']} v {row['away_team']}: "
                      f"model={prob:.1%}, "
                      f"odds={row['best_home_odds']:.2f}/{row['best_away_odds']:.2f}")
        return

    # Display value bets
    display = value_bets[[
        "match", "side", "model_prob", "implied_prob",
        "best_odds", "bookmaker", "edge", "kelly_stake",
    ]].copy()
    display["model_prob"] = display["model_prob"].map("{:.1%}".format)
    display["implied_prob"] = display["implied_prob"].map("{:.1%}".format)
    display["edge"] = display["edge"].map("{:+.1%}".format)
    display["kelly_stake"] = display["kelly_stake"].map("${:.2f}".format)

    print(f"\n{'='*70}")
    print(f"VALUE BETS FOUND: {len(value_bets)}")
    print(f"{'='*70}")
    print(tabulate(display, headers="keys", tablefmt="simple", showindex=False))

    # Log to tracker if requested
    if args.log:
        tracker = BetTracker()
        for _, bet in value_bets.iterrows():
            tracker.log_bet(
                date=bet["commence"],
                match=bet["match"],
                side=bet["side"],
                odds=bet["best_odds"],
                stake=bet["kelly_stake"],
                model_prob=bet["model_prob"],
                bookmaker=bet["bookmaker"],
            )
        print(f"\n{len(value_bets)} bets logged to tracker.")


if __name__ == "__main__":
    main()
