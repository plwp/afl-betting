#!/usr/bin/env python3
"""Footy tipping mode: predict winners for upcoming AFL matches."""

import argparse
import os
import joblib
import pandas as pd
from tabulate import tabulate

from config import FEATURE_PATH, MODEL_DIR, FEATURE_COLS
from features import build_current_match_features
from scanner import fetch_odds, parse_odds


def tips_from_odds(predictor, history_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Generate tips from live odds data (uses odds for market features only)."""
    rows = []
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
        X = pd.DataFrame([feats])[FEATURE_COLS]
        prob_home = float(predictor.predict_proba(X)[0, 1])
        prob_away = 1.0 - prob_home

        if prob_home >= prob_away:
            tip = row["home_team"]
            confidence = prob_home
        else:
            tip = row["away_team"]
            confidence = prob_away

        rows.append({
            "match": f"{row['home_team']} v {row['away_team']}",
            "tip": tip,
            "confidence": confidence,
            "home_prob": prob_home,
            "away_prob": prob_away,
            "commence": row["commence_time"],
        })

    return pd.DataFrame(rows)


def tips_manual(predictor, history_df: pd.DataFrame, matchups: list[tuple[str, str]]) -> pd.DataFrame:
    """Generate tips from manually specified matchups (no odds needed)."""
    rows = []
    for home, away in matchups:
        feats = build_current_match_features(
            history_df,
            home_team=home,
            away_team=away,
        )
        if feats is None:
            print(f"  Warning: no data for {home} v {away}, skipping")
            continue
        X = pd.DataFrame([feats])[FEATURE_COLS]
        prob_home = float(predictor.predict_proba(X)[0, 1])
        prob_away = 1.0 - prob_home

        if prob_home >= prob_away:
            tip = home
            confidence = prob_home
        else:
            tip = away
            confidence = prob_away

        rows.append({
            "match": f"{home} v {away}",
            "tip": tip,
            "confidence": confidence,
            "home_prob": prob_home,
            "away_prob": prob_away,
        })

    return pd.DataFrame(rows)


def display_tips(tips_df: pd.DataFrame):
    """Pretty-print the tips table."""
    if tips_df.empty:
        print("No tips generated.")
        return

    display = tips_df.copy()
    display["confidence"] = display["confidence"].map("{:.0%}".format)
    display["home_prob"] = display["home_prob"].map("{:.0%}".format)
    display["away_prob"] = display["away_prob"].map("{:.0%}".format)

    cols = ["match", "tip", "confidence", "home_prob", "away_prob"]
    if "commence" in display.columns:
        cols.append("commence")

    print(f"\n{'='*60}")
    print(f"  FOOTY TIPS ({len(tips_df)} matches)")
    print(f"{'='*60}")
    print(tabulate(
        display[cols],
        headers={"match": "Match", "tip": "Tip", "confidence": "Conf",
                 "home_prob": "Home%", "away_prob": "Away%", "commence": "Kickoff"},
        tablefmt="simple",
        showindex=False,
    ))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AFL Footy Tipping Bot",
        epilog="Examples:\n"
               "  python run_tips.py                    # Tips from live odds\n"
               "  python run_tips.py --match 'Sydney v Geelong'\n"
               "  python run_tips.py --match 'Carlton v Essendon' --match 'Sydney v GWS Giants'\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--match", action="append", metavar="'Home v Away'",
                        help="Manual matchup (can repeat). Format: 'Home Team v Away Team'")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh odds cache")
    args = parser.parse_args()

    # Load model
    model_path = os.path.join(MODEL_DIR, "model_bundle.pkl")
    if not os.path.exists(model_path):
        print("No trained model found. Run: python model.py")
        return

    predictor = joblib.load(model_path)

    # Load historical data
    if not os.path.exists(FEATURE_PATH):
        print("No feature matrix found. Run: python features.py")
        return
    history_df = pd.read_parquet(FEATURE_PATH)

    if args.match:
        # Manual matchups
        matchups = []
        for m in args.match:
            parts = [t.strip() for t in m.split(" v ")]
            if len(parts) != 2:
                print(f"  Bad format: '{m}' — use 'Home Team v Away Team'")
                continue
            matchups.append((parts[0], parts[1]))

        if not matchups:
            return

        tips_df = tips_manual(predictor, history_df, matchups)
    else:
        # Live odds
        try:
            events = fetch_odds(force_refresh=args.refresh)
        except ValueError as e:
            print(f"Error: {e}")
            print("Tip: use --match to enter matchups manually without an API key")
            return
        except Exception as e:
            print(f"API error: {e}")
            return

        if not events:
            print("No upcoming AFL events found.")
            print("Tip: use --match to enter matchups manually")
            return

        odds_df = parse_odds(events)
        tips_df = tips_from_odds(predictor, history_df, odds_df)

    display_tips(tips_df)


if __name__ == "__main__":
    main()
