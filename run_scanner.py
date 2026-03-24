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
    INITIAL_BANKROLL, EDGE_THRESHOLD, MAX_ODDS, MIN_MODEL_PROB,
    FAVOURITE_ONLY,
)
from features import build_current_match_features
from scanner import fetch_odds, parse_odds, scan_value_bets
from model import EnsemblePredictor  # noqa: F401 — needed for pickle
from tracker import BetTracker
from betfair import get_betfair_data
from squiggle import get_enhanced_squiggle_data, fetch_season_form


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
    parser.add_argument("--max-odds", type=float, default=MAX_ODDS,
                        help="Maximum odds to bet (default: 3.0)")
    parser.add_argument("--min-prob", type=float, default=MIN_MODEL_PROB,
                        help="Minimum model probability (default: 0.55)")
    parser.add_argument("--fav-only", action="store_true", default=FAVOURITE_ONLY,
                        help="Favourite-only strategy (default: True)")
    parser.add_argument("--no-fav-only", dest="fav_only", action="store_false",
                        help="Disable favourite-only strategy")
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

    # Fetch season form for hot-dog filter
    season_form = fetch_season_form(current_year)

    # Scan for value
    value_bets = scan_value_bets(
        odds_df, model_probs, args.bankroll, args.edge,
        max_odds=args.max_odds, min_model_prob=args.min_prob,
        favourite_only=args.fav_only, season_form=season_form,
    )

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

    # Split arbs from EV bets for display
    arb_bets = value_bets[value_bets.get("is_arb", False) == True] if "is_arb" in value_bets.columns else pd.DataFrame()
    ev_bets = value_bets[value_bets.get("is_arb", False) != True] if "is_arb" in value_bets.columns else value_bets

    if not arb_bets.empty:
        arb_display = arb_bets[[
            "match", "side", "best_odds", "bookmaker", "edge", "kelly_stake",
        ]].copy()
        arb_display["edge"] = arb_display["edge"].map("{:+.2%}".format)
        arb_display["kelly_stake"] = arb_display["kelly_stake"].map("${:.2f}".format)
        arb_display.rename(columns={"edge": "arb_margin"}, inplace=True)
        n_arb_matches = arb_bets["match"].nunique()
        profit = arb_bets.drop_duplicates("match")["arb_profit"].sum() if "arb_profit" in arb_bets.columns else 0
        print(f"\n{'='*70}")
        print(f"ARBITRAGE FOUND: {n_arb_matches} match(es), ${profit:.2f} guaranteed profit")
        print(f"{'='*70}")
        print(tabulate(arb_display, headers="keys", tablefmt="simple", showindex=False))

    if not ev_bets.empty:
        display = ev_bets[[
            "match", "side", "model_prob", "implied_prob",
            "best_odds", "bookmaker", "edge", "kelly_stake",
        ]].copy()
        display["model_prob"] = display["model_prob"].map("{:.1%}".format)
        display["implied_prob"] = display["implied_prob"].map("{:.1%}".format)
        display["edge"] = display["edge"].map("{:+.1%}".format)
        display["kelly_stake"] = display["kelly_stake"].map("${:.2f}".format)
        print(f"\n{'='*70}")
        print(f"VALUE BETS FOUND: {len(ev_bets)}")
        print(f"{'='*70}")
        print(tabulate(display, headers="keys", tablefmt="simple", showindex=False))

    if arb_bets.empty and ev_bets.empty:
        print(f"\n{'='*70}")
        print("No bets found.")
        print(f"{'='*70}")

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
