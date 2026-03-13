#!/usr/bin/env python3
"""Entry point: run walk-forward backtest."""

import os
import argparse
import pandas as pd
from data_ingest import run as ingest
from features import build_feature_matrix
from backtest import walk_forward_backtest, plot_bankroll
from config import MERGED_PATH, FEATURE_PATH, MODEL_DIR, EDGE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="AFL Walk-Forward Backtest")
    parser.add_argument("--start", type=int, default=2015, help="Start year")
    parser.add_argument("--end", type=int, default=2024, help="End year")
    parser.add_argument("--edge", type=float, default=EDGE_THRESHOLD,
                        help="Minimum edge threshold")
    parser.add_argument("--stacker", action="store_true",
                        help="Use logit stacker instead of fixed-weight blend")
    args = parser.parse_args()

    # Step 1: Ingest data if needed
    if not os.path.exists(MERGED_PATH):
        print("=== Data Ingestion ===")
        ingest()

    # Step 2: Build features if needed
    if not os.path.exists(FEATURE_PATH):
        print("\n=== Feature Engineering ===")
        build_feature_matrix()

    # Step 3: Run backtest
    mode = "stacker" if args.stacker else "fixed-weight 70/15/15"
    print(f"\n=== Walk-Forward Backtest ({mode}, edge={args.edge:.0%}) ===")
    df = pd.read_parquet(FEATURE_PATH)
    results = walk_forward_backtest(
        df,
        start_year=args.start,
        end_year=args.end,
        edge_threshold=args.edge,
        use_stacker=args.stacker,
    )

    # Step 4: Plot bankroll curve
    if results.get("bankroll_history"):
        os.makedirs(MODEL_DIR, exist_ok=True)
        plot_bankroll(results["bankroll_history"],
                      os.path.join(MODEL_DIR, "bankroll_curve.png"))

    # Step 5: Show yearly breakdown
    if "bets_df" in results and not results["bets_df"].empty:
        bets = results["bets_df"]
        print("\n=== Yearly Breakdown ===")
        yearly = bets.groupby("year").agg(
            bets=("pnl", "count"),
            pnl=("pnl", "sum"),
            staked=("stake", "sum"),
            win_rate=("won", "mean"),
        )
        yearly["yield"] = yearly["pnl"] / yearly["staked"]
        print(yearly.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
