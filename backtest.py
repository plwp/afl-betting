"""Walk-forward backtesting engine."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb

from config import (
    FEATURE_COLS, ELO_K, KELLY_FRACTION, MAX_BET_FRACTION,
    MIN_STAKE, EDGE_THRESHOLD, INITIAL_BANKROLL,
)
from sizing import kelly_stake, edge


def walk_forward_backtest(df: pd.DataFrame,
                          start_year: int = 2015,
                          end_year: int = 2024,
                          initial_bankroll: float = INITIAL_BANKROLL,
                          edge_threshold: float = EDGE_THRESHOLD) -> dict:
    """Walk-forward backtest with yearly retraining.

    For each year Y in [start_year, end_year]:
      - Train on all data up to Y-2
      - Calibrate on Y-1
      - Bet on Y

    Returns dict with results and bet log.
    """
    bankroll = initial_bankroll
    bet_log = []
    bankroll_history = [(df[df["year"] == start_year]["date"].min(), bankroll)]

    for year in range(start_year, end_year + 1):
        train_data = df[df["year"] <= year - 2]
        cal_data = df[df["year"] == year - 1]
        test_data = df[df["year"] == year]

        if len(train_data) < 100 or len(cal_data) < 20 or len(test_data) == 0:
            print(f"  Skipping {year}: insufficient data")
            continue

        X_train = train_data[FEATURE_COLS]
        y_train = train_data["home_win"].values
        X_cal = cal_data[FEATURE_COLS]
        y_cal = cal_data["home_win"].values

        # Train LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # Calibrate
        cal_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
        cal_model.fit(X_cal, y_cal)

        # Predict on test year
        X_test = test_data[FEATURE_COLS]
        probs = cal_model.predict_proba(X_test)[:, 1]

        year_bets = 0
        year_pnl = 0

        for i, (idx, row) in enumerate(test_data.iterrows()):
            prob_home = probs[i]
            prob_away = 1 - prob_home
            odds_home = row.get("odds_home", None)
            odds_away = row.get("odds_away", None)

            # Try home bet
            if odds_home is not None and not np.isnan(odds_home):
                e_home = edge(prob_home, odds_home)
                if e_home > edge_threshold:
                    stake = kelly_stake(prob_home, odds_home, bankroll)
                    if stake > 0:
                        won = row["home_win"] == 1
                        pnl = stake * (odds_home - 1) if won else -stake
                        bankroll += pnl
                        year_pnl += pnl
                        year_bets += 1
                        # CLV: closing line value
                        odds_close = row.get("odds_home_close", odds_home)
                        if pd.isna(odds_close):
                            odds_close = odds_home
                        clv = (1 / odds_close) - (1 / odds_home)
                        bet_log.append({
                            "date": row["date"],
                            "year": year,
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "side": "home",
                            "model_prob": prob_home,
                            "odds": odds_home,
                            "odds_close": odds_close,
                            "edge": e_home,
                            "stake": stake,
                            "won": won,
                            "pnl": pnl,
                            "clv": clv,
                            "bankroll": bankroll,
                        })
                        bankroll_history.append((row["date"], bankroll))

            # Try away bet
            if odds_away is not None and not np.isnan(odds_away):
                e_away = edge(prob_away, odds_away)
                if e_away > edge_threshold:
                    stake = kelly_stake(prob_away, odds_away, bankroll)
                    if stake > 0:
                        won = row["home_win"] == 0
                        pnl = stake * (odds_away - 1) if won else -stake
                        bankroll += pnl
                        year_pnl += pnl
                        year_bets += 1
                        odds_close = row.get("odds_away_close", odds_away)
                        if pd.isna(odds_close):
                            odds_close = odds_away
                        clv = (1 / odds_close) - (1 / odds_away)
                        bet_log.append({
                            "date": row["date"],
                            "year": year,
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "side": "away",
                            "model_prob": prob_away,
                            "odds": odds_away,
                            "odds_close": odds_close,
                            "edge": e_away,
                            "stake": stake,
                            "won": won,
                            "pnl": pnl,
                            "clv": clv,
                            "bankroll": bankroll,
                        })
                        bankroll_history.append((row["date"], bankroll))

        print(f"  {year}: {year_bets} bets, P&L ${year_pnl:+.2f}, "
              f"Bankroll ${bankroll:.2f}")

    bets_df = pd.DataFrame(bet_log)
    return _compute_summary(bets_df, bankroll_history, initial_bankroll)


def _compute_summary(bets_df: pd.DataFrame,
                     bankroll_history: list,
                     initial_bankroll: float) -> dict:
    """Compute backtest summary statistics."""
    if bets_df.empty:
        print("No bets placed.")
        return {"bets_df": bets_df, "bankroll_history": bankroll_history}

    total_staked = bets_df["stake"].sum()
    total_pnl = bets_df["pnl"].sum()
    n_bets = len(bets_df)
    n_wins = bets_df["won"].sum()
    win_rate = n_wins / n_bets

    roi = total_pnl / initial_bankroll
    yield_pct = total_pnl / total_staked if total_staked > 0 else 0
    avg_clv = bets_df["clv"].mean()

    # Max drawdown
    bankroll_series = pd.Series([b for _, b in bankroll_history])
    peak = bankroll_series.cummax()
    drawdown = (bankroll_series - peak) / peak
    max_dd = drawdown.min()

    # Sharpe-like ratio (daily returns)
    bets_df["return"] = bets_df["pnl"] / bets_df["stake"]
    daily_returns = bets_df.groupby("date")["return"].mean()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(len(daily_returns))
              if daily_returns.std() > 0 else 0)

    summary = {
        "total_bets": n_bets,
        "wins": int(n_wins),
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi": roi,
        "yield_pct": yield_pct,
        "avg_clv": avg_clv,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "final_bankroll": bankroll_history[-1][1],
    }

    print("\n=== Backtest Results ===")
    print(f"  Total Bets:     {n_bets}")
    print(f"  Win Rate:       {win_rate:.1%}")
    print(f"  Total Staked:   ${total_staked:,.2f}")
    print(f"  Total P&L:      ${total_pnl:+,.2f}")
    print(f"  ROI:            {roi:+.1%}")
    print(f"  Yield:          {yield_pct:+.1%}")
    print(f"  Avg CLV:        {avg_clv:+.4f}")
    print(f"  Max Drawdown:   {max_dd:.1%}")
    print(f"  Sharpe-like:    {sharpe:.2f}")
    print(f"  Final Bankroll: ${bankroll_history[-1][1]:,.2f}")

    return {
        "summary": summary,
        "bets_df": bets_df,
        "bankroll_history": bankroll_history,
    }


def plot_bankroll(bankroll_history: list, path: str = "models/bankroll_curve.png"):
    """Plot bankroll over time."""
    dates = [d for d, _ in bankroll_history]
    values = [v for _, v in bankroll_history]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, linewidth=1.5)
    ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Backtest Bankroll Curve")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Bankroll curve saved: {path}")
