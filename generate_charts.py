#!/usr/bin/env python3
"""Generate publication-quality charts for the README / technical note."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import FEATURE_COLS, FEATURE_PATH, MODEL_DIR
from model import train_models, temporal_split, _clip_probs, EnsemblePredictor
from backtest import walk_forward_backtest
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

df = pd.read_parquet(FEATURE_PATH)


# ── 1. Run backtest to get bet log ──────────────────────────────────────────
print("Running backtest...")
results = walk_forward_backtest(df, start_year=2015, end_year=2024)
bets = results["bets_df"]
bankroll_history = results["bankroll_history"]


# ── 2. Bankroll curve (improved) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
dates = [d for d, _ in bankroll_history]
values = [v for _, v in bankroll_history]
ax.plot(dates, values, linewidth=1.8, color="#2563eb")
ax.fill_between(dates, values[0], values, alpha=0.08, color="#2563eb")
ax.axhline(y=1000, color="#6b7280", linestyle="--", alpha=0.6, linewidth=0.8)
ax.set_ylabel("Bankroll ($)")
ax.set_title("Fig. 1: Bankroll Evolution (Walk-Forward Backtest, 2015-2024)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Annotate key points
final = values[-1]
ax.annotate(f"${final:,.0f}", xy=(dates[-1], final),
            xytext=(-60, 15), textcoords="offset points",
            fontsize=10, color="#2563eb",
            arrowprops=dict(arrowstyle="-", color="#2563eb", alpha=0.5))

peak_idx = np.argmax(values)
ax.annotate(f"Peak ${values[peak_idx]:,.0f}", xy=(dates[peak_idx], values[peak_idx]),
            xytext=(-70, 15), textcoords="offset points",
            fontsize=9, color="#16a34a",
            arrowprops=dict(arrowstyle="-", color="#16a34a", alpha=0.5))

trough_idx = np.argmin(values)
ax.annotate(f"Trough ${values[trough_idx]:,.0f}", xy=(dates[trough_idx], values[trough_idx]),
            xytext=(10, -20), textcoords="offset points",
            fontsize=9, color="#dc2626",
            arrowprops=dict(arrowstyle="-", color="#dc2626", alpha=0.5))

fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "bankroll_curve.png"), bbox_inches="tight")
plt.close(fig)
print("  bankroll_curve.png")


# ── 3. Year-by-year P&L bar chart ──────────────────────────────────────────
yearly = bets.groupby("year").agg(
    pnl=("pnl", "sum"),
    bets=("pnl", "count"),
    win_rate=("won", "mean"),
    staked=("stake", "sum"),
).reindex(range(2015, 2025), fill_value=0)
yearly["yield_pct"] = (yearly["pnl"] / yearly["staked"].replace(0, np.nan) * 100).fillna(0)

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = ["#16a34a" if v >= 0 else "#dc2626" for v in yearly["pnl"]]
bars = ax.bar(yearly.index, yearly["pnl"], color=colors, width=0.7, edgecolor="white", linewidth=0.5)

for bar, pnl, n in zip(bars, yearly["pnl"], yearly["bets"]):
    if n > 0:
        label = f"${pnl:+.0f}\n({int(n)} bets)"
    else:
        label = "0 bets"
    y_offset = 3 if pnl >= 0 else -3
    va = "bottom" if pnl >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, pnl + y_offset, label,
            ha="center", va=va, fontsize=8, color="#374151")

ax.axhline(y=0, color="#6b7280", linewidth=0.8)
ax.set_xlabel("Season")
ax.set_ylabel("Profit / Loss ($)")
ax.set_title("Fig. 2: Annual P&L (Favourite-Only Strategy, Quarter-Kelly)")
ax.set_xticks(range(2015, 2025))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:+,.0f}"))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "yearly_pnl.png"), bbox_inches="tight")
plt.close(fig)
print("  yearly_pnl.png")


# ── 4. Model comparison bar chart ───────────────────────────────────────────
print("Training model for evaluation metrics...")
result = train_models(df)
predictor = result["predictor"]
_, val, test = temporal_split(df)

X_test = test[FEATURE_COLS]
y_test = test["home_win"].to_numpy()
market_test = test["market_prob_home"].to_numpy()
lr_test, lgb_test, _ = predictor._base_probs(X_test)
ens_test = predictor.predict_proba(X_test)[:, 1]

models = ["Market", "LogReg", "LightGBM", "Ensemble"]
ll_values = [
    log_loss(y_test, _clip_probs(market_test)),
    log_loss(y_test, _clip_probs(lr_test)),
    log_loss(y_test, _clip_probs(lgb_test)),
    log_loss(y_test, _clip_probs(ens_test)),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
bar_colors = ["#6b7280", "#f59e0b", "#8b5cf6", "#2563eb"]
bars = ax.barh(models, ll_values, color=bar_colors, height=0.5, edgecolor="white")

# Add value labels
for bar, val in zip(bars, ll_values):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

ax.set_xlabel("Test Log Loss (lower is better)")
ax.set_title("Fig. 3: Model Comparison on Test Set (2023-2024)")
ax.set_xlim(0.585, 0.607)
ax.axvline(x=ll_values[0], color="#6b7280", linestyle="--", alpha=0.4, linewidth=0.8)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "model_comparison.png"), bbox_inches="tight")
plt.close(fig)
print("  model_comparison.png")


# ── 5. Calibration comparison (market vs ensemble, same plot) ───────────────
fig, ax = plt.subplots(figsize=(6, 6))

for probs, label, color, marker in [
    (market_test, "Market", "#6b7280", "o"),
    (ens_test, "Ensemble", "#2563eb", "s"),
]:
    frac_pos, mean_pred = calibration_curve(y_test, _clip_probs(probs), n_bins=10)
    ax.plot(mean_pred, frac_pos, f"{marker}-", label=label, color=color, markersize=6)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Observed Frequency")
ax.set_title("Fig. 4: Calibration Curves (Test Set)")
ax.legend(loc="upper left")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "calibration.png"), bbox_inches="tight")
plt.close(fig)
print("  calibration.png")


# ── 6. Feature importance (top 15, horizontal) ─────────────────────────────
importance = pd.Series(
    predictor.lgb_model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=True)
top15 = importance.tail(15)

fig, ax = plt.subplots(figsize=(8, 5.5))
colors_fi = ["#2563eb" if "market" in f or "elo" in f
             else "#16a34a" if "ewma" in f or "form" in f or "win_pct" in f
             else "#f59e0b" if "venue" in f or "travel" in f or "home_state" in f
             else "#8b5cf6" if "squiggle" in f
             else "#6b7280"
             for f in top15.index]
ax.barh(top15.index, top15.values, color=colors_fi, height=0.6)

for i, (feat, val) in enumerate(zip(top15.index, top15.values)):
    ax.text(val + 2, i, str(int(val)), va="center", fontsize=9)

ax.set_xlabel("Split Importance")
ax.set_title("Fig. 5: LightGBM Feature Importance (Top 15)")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2563eb", label="Market/Elo"),
    Patch(facecolor="#16a34a", label="Form/Performance"),
    Patch(facecolor="#f59e0b", label="Venue/Travel"),
    Patch(facecolor="#8b5cf6", label="External Models"),
    Patch(facecolor="#6b7280", label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), bbox_inches="tight")
plt.close(fig)
print("  feature_importance.png")


# ── 7. Cumulative P&L with bet markers ──────────────────────────────────────
bets_sorted = bets.sort_values("date").reset_index(drop=True)
bets_sorted["cum_pnl"] = bets_sorted["pnl"].cumsum()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.step(range(len(bets_sorted)), bets_sorted["cum_pnl"], where="post",
        linewidth=1.5, color="#2563eb")

wins = bets_sorted[bets_sorted["won"] == True]
losses = bets_sorted[bets_sorted["won"] == False]
ax.scatter(wins.index, wins["cum_pnl"], color="#16a34a", s=30, zorder=5,
           label=f"Win ({len(wins)})", edgecolors="white", linewidth=0.5)
ax.scatter(losses.index, losses["cum_pnl"], color="#dc2626", s=30, zorder=5,
           label=f"Loss ({len(losses)})", edgecolors="white", linewidth=0.5)

ax.axhline(y=0, color="#6b7280", linestyle="--", alpha=0.5, linewidth=0.8)
ax.set_xlabel("Bet Number")
ax.set_ylabel("Cumulative P&L ($)")
ax.set_title("Fig. 6: Cumulative P&L by Bet Sequence")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:+,.0f}"))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "cumulative_pnl.png"), bbox_inches="tight")
plt.close(fig)
print("  cumulative_pnl.png")


# ── 8. Edge distribution ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
win_edges = bets[bets["won"] == True]["edge"]
loss_edges = bets[bets["won"] == False]["edge"]

bins = np.linspace(0.05, bets["edge"].max() + 0.01, 12)
ax.hist(win_edges, bins=bins, alpha=0.7, color="#16a34a", label=f"Wins (n={len(win_edges)})", edgecolor="white")
ax.hist(loss_edges, bins=bins, alpha=0.7, color="#dc2626", label=f"Losses (n={len(loss_edges)})", edgecolor="white")
ax.axvline(x=bets["edge"].mean(), color="#2563eb", linestyle="--", linewidth=1.2,
           label=f"Mean edge: {bets['edge'].mean():.1%}")
ax.set_xlabel("Predicted Edge (model_prob * odds - 1)")
ax.set_ylabel("Count")
ax.set_title("Fig. 7: Distribution of Bet Edges (Wins vs Losses)")
ax.legend()
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "edge_distribution.png"), bbox_inches="tight")
plt.close(fig)
print("  edge_distribution.png")

print(f"\nAll charts saved to {CHARTS_DIR}/")
