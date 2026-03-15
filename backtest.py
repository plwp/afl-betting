"""Walk-forward backtesting engine."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    EDGE_THRESHOLD, EDGE_THRESHOLD_DOG, FEATURE_COLS, INITIAL_BANKROLL,
    MAX_ODDS, MIN_MODEL_PROB, MIN_STAKE,
)
from model import fit_model_bundle, _clip_probs, _tune_logreg, _tune_lgbm
from sizing import edge, kelly_stake
from strategy import BettingStrategy


def _select_bets(row: pd.Series, prob_home: float, bankroll: float, edge_threshold: float):
    """Return all sides with positive edge above threshold (bet both sides if value exists)."""
    bets = []

    odds_home = row.get("odds_home")
    if odds_home is not None and not pd.isna(odds_home):
        home_edge = edge(prob_home, odds_home)
        if home_edge > edge_threshold:
            stake = kelly_stake(prob_home, odds_home, bankroll)
            if stake > 0:
                odds_close = row.get("odds_home_close", odds_home)
                if pd.isna(odds_close):
                    odds_close = odds_home
                bets.append({
                    "side": "home", "model_prob": prob_home, "odds": odds_home,
                    "edge": home_edge, "stake": stake, "won": row["home_win"] == 1,
                    "odds_close": odds_close,
                })

    odds_away = row.get("odds_away")
    prob_away = 1.0 - prob_home
    if odds_away is not None and not pd.isna(odds_away):
        away_edge = edge(prob_away, odds_away)
        if away_edge > edge_threshold:
            stake = kelly_stake(prob_away, odds_away, bankroll)
            if stake > 0:
                odds_close = row.get("odds_away_close", odds_away)
                if pd.isna(odds_close):
                    odds_close = odds_away
                bets.append({
                    "side": "away", "model_prob": prob_away, "odds": odds_away,
                    "edge": away_edge, "stake": stake, "won": row["home_win"] == 0,
                    "odds_close": odds_close,
                })

    return bets


def _fixed_weight_probs(
    train_data: pd.DataFrame,
    cal_data: pd.DataFrame,
    test_data: pd.DataFrame,
    w_market: float = 0.7,
    w_lr: float = 0.15,
    w_lgb: float = 0.15,
) -> np.ndarray:
    """Compute blended probabilities with fixed weights (no stacker overfitting)."""
    X_train = train_data[FEATURE_COLS]
    y_train = train_data["home_win"].to_numpy()
    X_cal = cal_data[FEATURE_COLS]
    y_cal = cal_data["home_win"].to_numpy()
    X_test = test_data[FEATURE_COLS]

    scaler, logreg, _ = _tune_logreg(X_train, y_train, X_cal, y_cal)
    lgb_model, lgb_meta = _tune_lgbm(X_train, y_train, X_cal, y_cal)

    lr_probs = _clip_probs(logreg.predict_proba(scaler.transform(X_test))[:, 1])
    lgb_probs = _clip_probs(lgb_model.predict_proba(X_test)[:, 1])
    market_probs = _clip_probs(test_data["market_prob_home"].to_numpy())

    blended = _clip_probs(w_market * market_probs + w_lr * lr_probs + w_lgb * lgb_probs)
    return blended, lgb_meta


def walk_forward_backtest(
    df: pd.DataFrame,
    start_year: int = 2015,
    end_year: int = 2024,
    initial_bankroll: float = INITIAL_BANKROLL,
    edge_threshold: float = EDGE_THRESHOLD,
    edge_threshold_dog: float = EDGE_THRESHOLD_DOG,
    use_stacker: bool = True,
    max_odds: float = MAX_ODDS,
    min_model_prob: float = MIN_MODEL_PROB,
) -> dict:
    """Walk-forward backtest with yearly retraining, single bet per match, daily bankroll lock."""
    bankroll = float(initial_bankroll)
    bet_log = []
    bankroll_history = [(df[df["year"] == start_year]["date"].min(), bankroll)]

    strategy = BettingStrategy(
        edge_threshold_fav=edge_threshold,
        edge_threshold_dog=edge_threshold_dog,
        max_odds=max_odds,
        min_model_prob=min_model_prob,
    )

    for year in range(start_year, end_year + 1):
        train_data = df[df["year"] <= year - 3].copy()
        cal_data = df[(df["year"] >= year - 2) & (df["year"] <= year - 1)].copy()
        test_data = df[df["year"] == year].copy().sort_values("date", kind="mergesort")

        if len(train_data) < 100 or len(cal_data) < 40 or len(test_data) == 0:
            print(f"  Skipping {year}: insufficient data")
            continue

        if use_stacker:
            predictor, meta = fit_model_bundle(train_data, cal_data)
            probs = predictor.predict_proba(
                test_data[FEATURE_COLS],
            )[:, 1]
            # Margin predictions for line betting
            margin_preds = predictor.margin_model.predict(test_data[FEATURE_COLS])
            margin_std = predictor.margin_residual_std
        else:
            probs, lgb_meta = _fixed_weight_probs(train_data, cal_data, test_data)
            margin_preds = None
            margin_std = None

        print(
            f"  {year}: trained with {len(train_data)} train / {len(cal_data)} cal rows"
        )
        year_bets = 0
        year_pnl = 0.0

        # Daily bankroll lock: size bets off start-of-day bankroll
        current_date = None
        daily_start_bankroll = bankroll
        pending_pnl = 0.0

        for idx, row in test_data.reset_index(drop=True).iterrows():
            if bankroll < MIN_STAKE:
                print(f"  Stopping in {year}: bankroll ${bankroll:.2f} below minimum")
                break

            # Date change: apply previous day's P&L and reset
            if current_date is not None and row["date"] != current_date:
                bankroll += pending_pnl
                daily_start_bankroll = bankroll
                pending_pnl = 0.0
            current_date = row["date"]

            m_pred = float(margin_preds[idx]) if margin_preds is not None else None
            candidates = strategy.select_bets(
                row, float(probs[idx]), daily_start_bankroll,
                margin_pred=m_pred, margin_std=margin_std,
            )
            if not candidates:
                continue

            for candidate in candidates:
                pnl = candidate["stake"] * (candidate["odds"] - 1) if candidate["won"] else -candidate["stake"]
                pending_pnl += pnl
                year_pnl += pnl
                year_bets += 1

                clv = (1 / candidate["odds_close"]) - (1 / candidate["odds"])
                bet_log.append({
                    "date": row["date"],
                    "year": year,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "side": candidate["side"],
                    "model_prob": candidate["model_prob"],
                    "market_prob_home": row["market_prob_home"],
                    "odds": candidate["odds"],
                    "odds_close": candidate["odds_close"],
                    "edge": candidate["edge"],
                    "stake": candidate["stake"],
                    "won": candidate["won"],
                    "pnl": pnl,
                    "clv": clv,
                    "bankroll": bankroll + pending_pnl,
                })
                bankroll_history.append((row["date"], bankroll + pending_pnl))

        # End of year: apply remaining pending P&L
        bankroll += pending_pnl

        print(f"  {year}: {year_bets} bets, P&L ${year_pnl:+.2f}, Bankroll ${bankroll:.2f}")
        if bankroll < MIN_STAKE:
            break

    bets_df = pd.DataFrame(bet_log)
    return _compute_summary(bets_df, bankroll_history, initial_bankroll)


def _compute_summary(
    bets_df: pd.DataFrame,
    bankroll_history: list,
    initial_bankroll: float,
) -> dict:
    """Compute backtest summary statistics."""
    if bets_df.empty:
        print("No bets placed.")
        return {"bets_df": bets_df, "bankroll_history": bankroll_history}

    total_staked = float(bets_df["stake"].sum())
    total_pnl = float(bets_df["pnl"].sum())
    n_bets = int(len(bets_df))
    n_wins = int(bets_df["won"].sum())
    win_rate = n_wins / n_bets

    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    bankroll_return = (bankroll_history[-1][1] / initial_bankroll) - 1 if initial_bankroll else 0.0
    avg_clv = float(bets_df["clv"].mean())

    bankroll_series = pd.Series([b for _, b in bankroll_history], dtype=float)
    peak = bankroll_series.cummax()
    drawdown = (bankroll_series - peak) / peak.replace(0, np.nan)
    max_dd = float(drawdown.min())

    daily_pnl = bets_df.groupby("date")["pnl"].sum()
    sharpe = 0.0
    if len(daily_pnl) > 1 and daily_pnl.std(ddof=0) > 0:
        sharpe = float((daily_pnl.mean() / daily_pnl.std(ddof=0)) * np.sqrt(len(daily_pnl)))

    summary = {
        "total_bets": n_bets,
        "wins": n_wins,
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_pnl": total_pnl,
        "roi": roi,
        "bankroll_return": bankroll_return,
        "avg_clv": avg_clv,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "final_bankroll": bankroll_history[-1][1],
    }

    print("\n=== Backtest Results ===")
    print(f"  Total Bets:       {n_bets}")
    print(f"  Win Rate:         {win_rate:.1%}")
    print(f"  Total Staked:     ${total_staked:,.2f}")
    print(f"  Total P&L:        ${total_pnl:+,.2f}")
    print(f"  ROI on Stakes:    {roi:+.1%}")
    print(f"  Bankroll Return:  {bankroll_return:+.1%}")
    print(f"  Avg CLV:          {avg_clv:+.4f}")
    print(f"  Max Drawdown:     {max_dd:.1%}")
    print(f"  Sharpe-like:      {sharpe:.2f}")
    print(f"  Final Bankroll:   ${bankroll_history[-1][1]:,.2f}")

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
