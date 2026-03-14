#!/usr/bin/env python3
"""Entry point: scan live odds for value bets."""

import argparse
import os
import joblib
import pandas as pd
from tabulate import tabulate

from datetime import datetime

from config import (
    FEATURE_PATH, MODEL_DIR, FEATURE_COLS,
    INITIAL_BANKROLL, EDGE_THRESHOLD,
)
from features import build_current_match_features
from scanner import fetch_odds, parse_odds, scan_value_bets
from model import EnsemblePredictor  # noqa: F401 — needed for pickle
from tracker import BetTracker
from betfair import get_betfair_data
from squiggle import get_enhanced_squiggle_data


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
    model_path = os.path.join(MODEL_DIR, "model_bundle.pkl")
    if not os.path.exists(model_path):
        print("No trained model found. Run model.py or run_backtest.py first.")
        return

    predictor = joblib.load(model_path)

    # Load feature matrix for historical data
    if not os.path.exists(FEATURE_PATH):
        print("No feature matrix found. Run features.py first.")
        return
    history_df = pd.read_parquet(FEATURE_PATH)

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

    # Enrich with Betfair Exchange data
    betfair_data = get_betfair_data()

    # Enrich with enhanced Squiggle data (current year + estimate round)
    now = datetime.now()
    current_year = now.year
    # Estimate current round from week of year (AFL starts ~mid-March)
    week = now.isocalendar()[1]
    estimated_round = max(1, week - 10)
    squiggle_data = get_enhanced_squiggle_data(current_year, estimated_round)

    # Build model predictions for each matchup
    model_probs = {}
    for _, row in odds_df.iterrows():
        feats = build_current_match_features(
            history_df,
            home_team=row["home_team"],
            away_team=row["away_team"],
            odds_home=row["best_home_odds"],
            odds_away=row["best_away_odds"],
        )
        if feats is None:
            continue

        # Override with live Betfair data if available
        key = (row["home_team"], row["away_team"])
        if key in betfair_data:
            bf = betfair_data[key]
            feats["bf_spread_home"] = bf["bf_spread_home"]
            feats["bf_spread_away"] = bf["bf_spread_away"]
            feats["bf_volume_ratio"] = bf["bf_volume_ratio"]

        # Override with live enhanced Squiggle data if available
        if key in squiggle_data:
            sq = squiggle_data[key]
            feats["squiggle_top3_prob"] = sq["squiggle_top3_prob"]
            feats["squiggle_model_spread"] = sq["squiggle_model_spread"]

        X = pd.DataFrame([feats])[FEATURE_COLS]
        prob = predictor.predict_proba(X)[0, 1]
        model_probs[(row["home_team"], row["away_team"])] = float(prob)

    # Scan for value
    value_bets = scan_value_bets(odds_df, model_probs, args.bankroll, args.edge)

    if value_bets.empty:
        print("\nNo value bets found above edge threshold.")
        print("\nAll matches:")
        for _, row in odds_df.iterrows():
            key = (row["home_team"], row["away_team"])
            prob = model_probs.get(key)
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
