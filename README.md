# Can Machine Learning Beat AFL Betting Markets?

**A failed experiment -- published as a learning resource.**

## Abstract

We built an ensemble machine learning system to identify value bets in Australian Rules Football (AFL) markets. The system combines Elo ratings, rolling performance statistics, weather data, and consensus model predictions, blended with bookmaker odds via a calibrated logit-space stacker. Over a 10-year walk-forward backtest (2015-2024), the strategy produced +8.4% ROI on 43 bets with a 67.4% win rate. However, the sample size is far too small for statistical significance, closing line value is negative (-0.008), and individual models fail to beat the market on log loss. We conclude that the results are consistent with variance rather than genuine edge, and publish the full system as a reference implementation.

## 1. Introduction

Sports betting markets are widely considered semi-efficient: bookmaker odds incorporate substantial information and are difficult to beat systematically. The AFL, with ~200 matches per season and a well-developed betting market in Australia, presents an interesting test case. The question is simple: can a quantitative model, trained on 16 years of historical data and enriched with external signals, find exploitable inefficiencies?

**Answer: not convincingly.** This document reports what we tried, what the numbers show, and why we believe the positive backtest results are likely noise.

## 2. Data

**Coverage**: 16 seasons (2009-2024), ~3,100 matches with opening and closing odds.

| Source | Data | Usage |
|--------|------|-------|
| AFL-Data-Analysis (GitHub) | Match results, scores, venues | Core match data |
| AusportsBetting.com | Historical bookmaker odds | Market probabilities, CLV |
| Squiggle API | ~20 public computer model predictions | Consensus signal |
| Open-Meteo API | Historical weather per venue | Rain, wind features |
| FootyWire | Team-level match statistics | Tested, excluded |
| Betfair Exchange | Back/lay spreads, matched volume | Live scanning only |
| The Odds API | Live odds from 8+ AU bookmakers | Live scanning only |

## 3. Methodology

### 3.1 Feature Engineering

40 features across 8 categories, all computed without lookahead bias (shifted/lagged appropriately):

| Category | Features | Construction |
|----------|----------|-------------|
| Elo ratings | `elo_diff`, `elo_prob` | Margin-based K multiplier (capped 2.5x), 30pt home advantage, 33% season reversion to mean |
| Market odds | `market_prob_home/away`, `overround`, `market_elo_delta` | Implied probabilities from opening odds, normalised for overround |
| Form | `form_*_5`, `win_pct_*_10`, `margin_ewma_*`, `scoring_ewma_*` | Rolling 5-game win rate, 10-game win%, EWMA (span=10) of margins and scores |
| Venue/travel | `venue_exp_*`, `is_home_state`, `travel_hours_*` | Cumulative venue appearances, interstate flight hours |
| Rest | `rest_days_*`, `rest_diff` | Days since last match (capped at 30) |
| Matchup | `h2h_home_win_pct` | Historical head-to-head win rate |
| Weather | `rain_mm`, `wind_speed`, `is_wet`, `is_roofed` | Open-Meteo data matched to venue coordinates |
| Squiggle | `squiggle_prob_home`, `top3_prob`, `model_spread` | Consensus of public models, top-3 accuracy-weighted, inter-model disagreement |

**Excluded features**: FootyWire team statistics (disposals, clearances, inside 50s, tackles) were tested as rolling 5-game EWMA features but reduced accuracy by 0.4% and were removed. Betfair Exchange features use neutral defaults in historical data and only activate during live scanning.

### 3.2 Model Architecture

```mermaid
graph LR
    A[40 Features] --> B[Logistic Regression]
    A --> C[LightGBM]
    D[Market Odds] --> E[Calibrated Stacker]
    B -->|logit prob| E
    C -->|logit prob| E
    B -->|LR - market delta| E
    C -->|LGB - market delta| E
    D -->|logit prob| E
    E --> F{Edge > 5%?}
    F -->|Yes| G[Kelly Stake]
    F -->|No| H[No Bet]
```

**Base models**:
- **Logistic Regression**: L2-regularised (C tuned from 0.02-4.0), scaled features
- **LightGBM**: Conservative configuration (300-500 trees, max depth 3-5, heavy L1/L2 regularisation, early stopping at 50 rounds)

**Stacker**: A logistic regression in logit space over 5 inputs -- `logit(LR)`, `logit(LGB)`, `logit(market)`, plus two delta features (`LGB - market`, `LR - market`). Tuned with C in [0.01, 0.1]. This learns how much to trust each signal; in practice it weights market odds at ~70%.

### 3.3 Betting Strategy

**Favourite-only** with strict filters:
- Model probability > 55%
- Market agrees it's the favourite (implied prob > 50%)
- Decimal odds <= 3.0
- Edge > 5%, where edge = model_prob x odds - 1
- At most one bet per match (highest edge side)
- **Quarter-Kelly** sizing (f* x 0.25), capped at 5% of bankroll

This replaced an earlier strategy that bet both sides and underdogs, which lost -$181. The switch to favourite-only flipped the backtest to +$110.

### 3.4 Walk-Forward Protocol

For each test year Y (2015-2024):
1. **Train** on all matches in years <= Y-3
2. **Calibrate** stacker on years Y-2 to Y-1
3. **Test** on year Y (out-of-sample)
4. **Daily bankroll lock**: bet sizes computed from start-of-day bankroll; P&L applied at end of day

This ensures no lookahead bias: the model for 2024 has never seen data from 2022 onwards.

## 4. Results

### 4.1 Model Evaluation

Static evaluation: trained on 2009-2020, calibrated on 2021-2022, tested on 2023-2024 (n=432 matches).

| Model | Log Loss | Brier Score | Accuracy | vs Market LL |
|-------|----------|-------------|----------|--------------|
| Market | 0.5929 | 0.2050 | 65.1% | -- |
| Logistic Regression | 0.5990 | 0.2077 | 65.5% | -0.0061 (worse) |
| LightGBM | 0.6023 | 0.2083 | 68.3% | -0.0094 (worse) |
| **Ensemble (stacker)** | **0.5916** | **0.2041** | **66.2%** | **+0.0012 (better)** |

Neither base model beats the market on log loss individually. The ensemble stacker recovers a marginal edge (+0.0012) by blending model signals with market odds.

![Model Comparison](charts/model_comparison.png)

### 4.2 Calibration

Both market and ensemble produce well-calibrated probabilities, tracking the diagonal closely. The ensemble shows slight underconfidence in the 40-60% range and overconfidence above 90%.

![Calibration](charts/calibration.png)

### 4.3 Feature Importance

Market-derived features dominate (top 3 are all market odds). The model's marginal contribution comes from venue experience, recent scoring trends, and weather -- signals the market may partially discount.

![Feature Importance](charts/feature_importance.png)

### 4.4 Backtest Performance

| Metric | Value |
|--------|-------|
| Total Bets | 43 |
| Win Rate | 67.4% (29W / 14L) |
| Total Staked | $1,312.78 |
| Total P&L | +$110.17 |
| ROI on Stakes | +8.4% |
| Bankroll Return | +11.0% ($1,000 -> $1,110) |
| Max Drawdown | -10.9% |
| Sharpe-like Ratio | 0.71 |
| Avg Closing Line Value | -0.0080 |

![Bankroll Curve](charts/bankroll_curve.png)

### 4.5 Annual Breakdown

| Year | Bets | Win Rate | P&L | Yield |
|------|------|----------|-----|-------|
| 2015 | 8 | 75% | +$19.56 | +7% |
| 2016 | 5 | 60% | -$12.86 | -10% |
| 2017 | 2 | 50% | -$10.57 | -23% |
| 2018 | 3 | 67% | +$7.58 | +12% |
| 2019 | 8 | 62% | +$14.77 | +6% |
| 2020 | 0 | -- | -- | -- |
| 2021 | 2 | 100% | +$27.99 | +56% |
| 2022 | 1 | 100% | +$17.72 | +53% |
| 2023 | 6 | 67% | +$24.15 | +13% |
| 2024 | 8 | 62% | +$21.83 | +8% |

![Annual P&L](charts/yearly_pnl.png)

### 4.6 Bet-Level Analysis

The cumulative P&L curve shows high path-dependency. The system spent bets 10-25 underwater, and the recovery is driven by a streak of wins from bet 25 onwards (2021-2024). Most bets cluster near the minimum 5% edge threshold.

![Cumulative P&L](charts/cumulative_pnl.png)

![Edge Distribution](charts/edge_distribution.png)

## 5. Discussion

### Why we think this doesn't work

1. **Insufficient sample size.** 43 bets over 10 years cannot establish statistical significance. A binomial test on the 67.4% win rate at the observed average odds gives p ~ 0.15 -- nowhere near the 0.05 threshold. You would need ~200+ bets at this win rate to reach significance.

2. **Negative closing line value.** The average CLV of -0.008 means the model is betting into lines that move against it. In efficient markets, positive CLV is the hallmark of a genuine edge. Negative CLV suggests the market is smarter than the model.

3. **Individual models lose to the market.** Both logistic regression and LightGBM have worse log loss than simply using bookmaker odds. The stacker recovers a tiny edge (+0.0012 log loss) by learning to mostly trust the market and nudge predictions slightly -- but this is a razor-thin margin.

4. **Small-sample flattery.** The 2021-2022 period (3 bets, 100% win rate, +$46) accounts for ~40% of total profit. Remove those 3 bets and ROI drops to ~5%.

5. **No live validation.** All results are backtested against historical odds. Real-world execution faces additional headwinds: odds may not be available at the backtested price, accounts may be limited, and the model has never been tested in production.

### What we learned

- **Markets are good.** The bookmaker line is the single strongest predictor. Three of the top four LightGBM features are market-derived. Any model that doesn't incorporate market odds performs substantially worse.
- **Stacking helps.** Even though base models lose to the market individually, the logit-space stacker can blend them in a way that marginally improves on the market alone. The key is the delta features (model - market) which capture where the model disagrees.
- **Strategy matters more than model.** Switching from "bet anything with edge" to "favourite-only with strict filters" flipped the backtest from -$181 to +$110 -- a larger effect than any modelling improvement.
- **More data doesn't always help.** FootyWire team statistics (disposals, clearances, etc.) added noise and hurt performance. Weather and Squiggle consensus helped marginally.
- **Quarter-Kelly is conservative enough.** The 0.25 Kelly fraction with a 5% bankroll cap kept maximum drawdown under 11%, even through losing streaks.

## 6. System Architecture

```mermaid
graph TD
    A[AFL Match CSVs<br/>2009-2024] --> D[data_ingest.py]
    B[AusportsBetting<br/>Historical Odds] --> D
    D --> E[afl_merged.parquet]
    E --> F[features.py]
    G[Open-Meteo API<br/>Weather] --> F
    H[Squiggle API<br/>Model Consensus] --> F
    I[FootyWire<br/>Team Stats] --> F
    F --> J[feature_matrix.parquet<br/>40 features per match]
    J --> K[model.py<br/>Train Ensemble]
    J --> L[backtest.py<br/>Walk-Forward Test]
    K --> M[model_bundle.pkl]
    M --> N[run_scanner.py<br/>Live Betting]
    O[The Odds API<br/>Live Odds] --> N
    P[Betfair Exchange<br/>Live Spreads] --> N
```

## 7. Reproduction

```bash
pip install -r requirements.txt

# Full pipeline: ingest -> features -> backtest
python run_backtest.py

# Train model and print evaluation metrics
python model.py

# Regenerate all charts
python generate_charts.py

# Live value bet scanner (requires ODDS_API_KEY in .env)
python run_scanner.py --bankroll 1000

# Cross-bookie arbitrage scanner
python run_arb_scanner.py --bankroll 1000

# Footy tipping predictions
python run_tips.py
```

### Project Structure

```
config.py              Configuration, feature columns, team/venue mappings
data_ingest.py         Download and merge match + odds data
features.py            Feature engineering (Elo, rolling stats, weather, etc.)
model.py               Ensemble training (LogReg + LightGBM + stacker)
backtest.py            Walk-forward backtesting engine
strategy.py            Favourite-only bet selection
sizing.py              Kelly criterion stake sizing
generate_charts.py     Publication charts for this README
squiggle.py            Squiggle API consensus predictions
weather.py             Open-Meteo weather data
betfair.py             Betfair Exchange API
team_stats.py          FootyWire scraper
tracker.py             SQLite bet tracking
run_backtest.py        Backtest entry point
run_scanner.py         Live value bet scanner
run_arb_scanner.py     Arbitrage scanner
run_tips.py            Tipping predictions
run_report.py          Performance reporting
```

## 8. Evolution

This project went through several iterations, each attempting to improve on the last:

1. **v1** -- Basic Elo + logistic regression. Bet both sides. Lost money.
2. **v2** (Gemini review) -- Margin-based Elo, ensemble model, daily bankroll lock. Still lost money.
3. **v3** (Codex review) -- Logit-space stacker, feature leakage fixes. Broke even.
4. **v4** -- Added weather, Squiggle consensus, FootyWire stats. Weather and Squiggle helped; team stats hurt.
5. **v5** -- Favourite-only strategy with tight filters. Flipped from -$181 to +$110.
6. **v6** -- Betfair Exchange integration (live only), enhanced Squiggle features. Marginal improvement.

The biggest single improvement came not from better modelling but from better strategy (v5).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
