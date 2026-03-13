#!/usr/bin/env python3
"""Entry point: run walk-forward backtest."""

import os
from data_ingest import run as ingest
from features import build_feature_matrix
from backtest import walk_forward_backtest, plot_bankroll
from config import MERGED_PATH, FEATURE_PATH, MODEL_DIR


def main():
    # Step 1: Ingest data if needed
    if not os.path.exists(MERGED_PATH):
        print("=== Data Ingestion ===")
        ingest()

    # Step 2: Build features if needed
    if not os.path.exists(FEATURE_PATH):
        print("\n=== Feature Engineering ===")
        build_feature_matrix()

    # Step 3: Run backtest
    print("\n=== Walk-Forward Backtest ===")
    import pandas as pd
    df = pd.read_parquet(FEATURE_PATH)
    results = walk_forward_backtest(df)

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
