# Can Machine Learning Beat AFL Betting Markets?

**A failed experiment -- published as a learning resource.**

## Abstract

We built an ensemble machine learning system to identify value bets in Australian Rules Football (AFL) markets. The system combines Elo ratings, Glicko-2 ratings, rolling performance statistics, weather data, and consensus model predictions, blended with bookmaker odds via a calibrated logit-space stacker. Over a 10-year walk-forward backtest (2015-2024), the strategy produced +12.8% ROI on 74 bets with a 67.6% win rate. However, the sample size is far too small for statistical significance, closing line value is negative (-0.009), and individual models fail to beat the market on log loss. We conclude that the results are consistent with variance rather than genuine edge, and publish the full system as a reference implementation.

**Version note (v10, 2026-03-15)**: Added a situational "home hot-dog" filter for selective underdog betting: home underdogs with hot form (>=40% of last 5) and close scoring (EWMA gap <=10 points) now qualify at 5% edge threshold. This adds 6 underdog bets to the 68 favourite bets, improving ROI from `+8.8%` to `+12.8%` ($1,335 final bankroll). Also investigated line/handicap betting using the margin regressor vs bookmaker spreads -- the book's margin RMSE (33.2) beats our model's (34.4), so line betting is not viable. Odds drift features (open-to-close movement) were tested as model inputs but hurt walk-forward performance. Line data is now ingested but used only for analysis.

## 1. Introduction

Sports betting markets are widely considered semi-efficient: bookmaker odds incorporate substantial information and are difficult to beat systematically. The AFL, with ~200 matches per season and a well-developed betting market in Australia, presents an interesting test case. The question is simple: can a quantitative model, trained on 16 years of historical data and enriched with external signals, find exploitable inefficiencies?

**Answer: not convincingly.** This document reports what we tried, what the numbers show, and why we believe the positive backtest results are likely noise.

## 2. Data

**Coverage**: 16 seasons (2009-2024), ~3,100 matches with opening and closing odds.

**Odds source**: Historical odds from AusportsBetting.com. These are "best available" market odds rather than odds from a single bookmaker, which means the backtested prices may not have been available from any single account in practice. This flatters ROI -- a real bettor would face worse prices on average.

| Source | Data | Usage |
|--------|------|-------|
| AFL-Data-Analysis (GitHub) | Match results, scores, venues | Core match data |
| AusportsBetting.com | Historical odds, handicap lines | Market probabilities, CLV, line analysis |
| Squiggle API | ~20 public computer model predictions | Consensus signal |
| Open-Meteo API | Historical weather per venue | Rain, wind features |
| FootyWire | Team-level match statistics | Tested, excluded |
| Betfair Exchange | Back/lay spreads, matched volume | Live scanning only |
| The Odds API | Live odds from 8+ AU bookmakers | Live scanning only |

## 3. Methodology

### 3.1 Feature Engineering

45 features across 9 categories, all computed without lookahead bias (shifted/lagged appropriately):

| Category | Features | Construction |
|----------|----------|-------------|
| Elo ratings | `elo_diff` | Margin-based K multiplier (capped 2.5x), 30pt home advantage, 33% season reversion to mean |
| Glicko-2 | `glicko_prob`, `glicko_uncertainty` | Full Glicko-2 with volatility; combined RD as uncertainty signal |
| Market odds | `market_prob_home/away`, `overround`, `market_elo_delta` | Implied probabilities from opening odds, normalised for overround |
| Form | `form_*_5`, `win_pct_*_10`, `margin_ewma_*`, `scoring_ewma_*` | Rolling 5-game win rate, 10-game win%, EWMA (span=10) of margins and scores |
| Volatility/momentum | `scoring_vol_*`, `form_accel_*`, `margin_trend_*` | Scoring std dev (last 10), form improvement (5-game vs 10-game), margin slope (last 5) |
| Venue/travel | `venue_exp_*`, `travel_hours_away` | Cumulative venue appearances, away team interstate flight hours |
| Rest | `rest_days_*`, `rest_diff` | Days since last match (capped at 30) |
| Matchup | `h2h_home_win_pct`, `rivalry_intensity`, `home_venue_pct`, `team_h2h_margin_ewma` | Historical head-to-head win rate, rolling abs margin between pair, venue familiarity, directional H2H EWMA |
| Weather | `rain_mm`, `wind_speed` | Open-Meteo data matched to venue coordinates |
| Squiggle | `squiggle_prob_home`, `top3_prob`, `model_spread` | Consensus of public models, top-3 accuracy-weighted, inter-model disagreement |

**Pruned features**: 11 features with zero LightGBM importance were removed in v7: `elo_prob` (redundant with `elo_diff`), `is_home_state`, `travel_hours_home`, `is_final`, `is_wet`, `is_roofed`, `bf_spread_home/away`, `bf_volume_ratio`, `same_state_derby`, `away_in_home_state`.

**v9 additions**: Three new feature families (9 features) were added after empirical analysis of underdog upset predictors: scoring volatility captures high-variance teams underpriced at long odds; form acceleration identifies teams improving faster than their record suggests; margin trend detects whether a team is winning by increasing or decreasing margins. All three improved walk-forward ROI.

**Training weights**: Exponential sample weighting with a 3-year half-life downweights older seasons, improving the model's adaptation to evolving league dynamics. A game from 3 years ago gets half the weight of a current-season game.

**Excluded features**: FootyWire team statistics, scoring shot conversion rates, ladder position, odds drift (open-to-close implied probability change), line movement, and ground dimensions were all tested but hurt walk-forward performance. A neural network with team embeddings was also tested and removed.

### 3.2 Model Architecture

```mermaid
graph LR
    A[45 Features] --> B[Logistic Regression]
    A --> C[LightGBM]
    A --> X[XGBoost]
    A --> M[Margin Regressor]
    D[Market Odds] --> E[Calibrated Stacker]
    B -->|logit prob| E
    C -->|logit prob| E
    X -->|logit prob| E
    M -->|logit prob| E
    D -->|logit prob| E
    B -->|LR - market delta| E
    C -->|LGB - market delta| E
    X -->|XGB - market delta| E
    M -->|Margin - market delta| E
    E --> F{Edge > 5%?}
    F -->|Yes| G[Kelly Stake]
    F -->|No| H[No Bet]
    F -->|Underdog?| I{Home Hot-Dog?}
    I -->|Yes| G
    I -->|No| H
```

**Base models**:
- **Logistic Regression**: L2-regularised (C tuned from 0.02-4.0), scaled features
- **LightGBM**: Conservative configuration (300-500 trees, max depth 3-5, heavy L1/L2 regularisation, early stopping at 50 rounds)
- **XGBoost**: Similar conservative configuration to LightGBM
- **Margin Regressor**: XGBoost regression on match margin, converted to win probability via Gaussian CDF

**Stacker**: A logistic regression in logit space over 11 inputs -- `logit(LR)`, `logit(LGB)`, `logit(XGB)`, `logit(margin)`, `logit(market)`, four delta features (model - market), plus `glicko_uncertainty` and `margin_confidence`. Tuned with C in [0.01, 0.1]. This learns how much to trust each signal; in practice it weights market odds at ~70%.

### 3.3 Betting Strategy

**Favourites**: standard filters apply:
- Model probability > 55%
- Market agrees it's the favourite (implied prob > 50%)
- Decimal odds <= 3.0
- Edge > 5%, where edge = model_prob x odds - 1

**Underdogs**: must pass the "home hot-dog" trifecta:
- Must be the home team (away dogs never qualify)
- Recent form >= 40% (won 2+ of last 5)
- Scoring EWMA within 10 points of opponent
- Edge > 5%

**Common rules**:
- At most one bet per match (highest edge side)
- **Quarter-Kelly** sizing (f* x 0.25), capped at 5% of bankroll

The strategy evolved from earlier approaches: v5 introduced favourite-only filters which flipped the backtest from -$181 to positive. v10 added the hot-dog filter after analysis showed home underdogs with hot form and close scoring had a historically +10% upset ROI.

**Important caveat**: these filters were chosen *after* observing backtest results. While model weights are out-of-sample (walk-forward retraining), the strategy itself was effectively fit to the 2015-2024 backtest period. This is a form of selection bias that likely inflates the reported ROI.

**Line betting (investigated, rejected)**: We tested using the margin regressor to bet against bookmaker handicap lines. The bookmaker's margin RMSE (33.2 points) beats our model's (34.4), and when we disagree with the book's line by >5 points, we're wrong more often than right (48.4%). Line betting at ~1.91 odds requires >52.4% accuracy to overcome the juice -- our model cannot deliver this on spread predictions.

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
| Logistic Regression | 0.5952 | 0.2066 | 64.8% | -0.0024 (worse) |
| LightGBM | 0.5957 | 0.2056 | 66.2% | -0.0029 (worse) |
| XGBoost | 0.5900 | 0.2036 | 66.2% | +0.0029 (better) |
| Margin Regressor | 0.5931 | 0.2041 | 65.7% | -0.0003 (tied) |
| **Ensemble (stacker)** | **0.5888** | **0.2031** | **66.7%** | **+0.0041 (better)** |

No base model beats the market on log loss individually. The ensemble stacker recovers a small edge (+0.0034) by blending model signals with market odds.

![Model Comparison](charts/model_comparison.png)

### 4.2 Calibration

Both market and ensemble produce well-calibrated probabilities, tracking the diagonal closely. The ensemble shows slight underconfidence in the 40-60% range and overconfidence above 90%.

![Calibration](charts/calibration.png)

### 4.3 Feature Importance

Market-derived features dominate (top 3 are all market odds). The model's marginal contribution comes from venue experience, Glicko-2 ratings, H2H matchup history, and weather -- signals the market may partially discount.

![Feature Importance](charts/feature_importance.png)

### 4.4 Backtest Performance

| Metric | Value |
|--------|-------|
| Total Bets | 74 |
| Win Rate | 67.6% (50W / 24L) |
| Total Staked | $2,624.75 |
| Total P&L | +$334.80 |
| ROI on Stakes | +12.8% |
| Bankroll Return | +33.5% ($1,000 -> $1,335) |
| Max Drawdown | -18.4% |
| Sharpe-like Ratio | 1.19 |
| Avg CLV (implied prob delta) | -0.0089 |

![Bankroll Curve](charts/bankroll_curve.png)

### 4.5 Annual Breakdown

| Year | Bets | Win Rate | P&L | Yield |
|------|------|----------|-----|-------|
| 2015 | 8 | 88% | +$63.53 | +27% |
| 2016 | 7 | 43% | -$7.14 | -3% |
| 2017 | 8 | 50% | -$81.91 | -27% |
| 2018 | 10 | 70% | +$74.70 | +23% |
| 2019 | 7 | 71% | +$76.30 | +41% |
| 2020 | 0 | -- | $0.00 | -- |
| 2021 | 10 | 90% | +$126.13 | +42% |
| 2022 | 4 | 50% | -$14.33 | -12% |
| 2023 | 2 | 50% | +$32.61 | +44% |
| 2024 | 18 | 67% | +$64.92 | +8% |

![Annual P&L](charts/yearly_pnl.png)

### 4.6 Bet-Level Analysis

The cumulative P&L curve shows high path-dependency. The 2021 season (10 bets, 90% win rate, +$126) is the strongest single year, but the strategy is profitable across 7 of 9 active years. The hot-dog filter adds selective underdog bets that contribute to diversification.

![Cumulative P&L](charts/cumulative_pnl.png)

![Edge Distribution](charts/edge_distribution.png)

## 5. Discussion

### Why we think this doesn't work

1. **Insufficient sample size.** 74 bets over 10 years cannot establish statistical significance. A binomial test on the 67.6% win rate at the observed average odds gives p ~ 0.07. You would need ~150+ bets at this win rate to reach significance.

2. **Negative closing line value.** The average CLV of -0.0089 (measured as implied probability delta: `1/odds_close - 1/odds_open`) means the model is betting into lines that move against it. In efficient markets, positive CLV is the hallmark of a genuine edge. Negative CLV suggests the market is smarter than the model and the opening price was already too generous.

3. **Individual models lose to the market.** All four base models have worse log loss than simply using bookmaker odds. The stacker recovers a small edge by learning to mostly trust the market and nudge predictions slightly -- but this is a razor-thin margin.

4. **Concentration risk.** The 2021 season (10 bets, 90% win rate, +$126) accounts for ~38% of total profit. The strategy is profitable without 2021, but margins are thin.

5. **No live validation.** All results are backtested against historical odds. Real-world execution faces additional headwinds: odds may not be available at the backtested price, accounts may be limited, and the model has never been tested in production.

6. **Strategy overfitting.** The favourite-only and hot-dog filters were discovered by iterating on the backtest. While model weights are genuinely out-of-sample, the decision to restrict to favourites with >55% model probability and odds <= 3.0, plus the hot-dog trifecta, was made after seeing that broader strategies lost money. This is a form of data mining that likely inflates the apparent edge.

7. **Favourite-longshot bias.** AFL markets typically have lower overround on favourites than on underdogs. By primarily betting favourites, the strategy may simply be paying less "tax" to the bookmaker rather than exploiting a genuine informational edge. The +12.8% ROI should be compared against a naive "bet all favourites" baseline, not zero.

8. **Line market efficiency.** We tested using our margin regressor to bet handicap lines and found the bookmaker's spread predictions are more accurate than ours (RMSE 33.2 vs 34.4). When our model disagrees with the book by >5 points, the book is right more often. This confirms that our edge, if any, lies in blending probability signals rather than outpredicting the market on margins.

### What we learned

- **Markets are good.** The bookmaker line is the single strongest predictor. Three of the top four LightGBM features are market-derived. Any model that doesn't incorporate market odds performs substantially worse.
- **Stacking helps.** Even though base models lose to the market individually, the logit-space stacker can blend them in a way that marginally improves on the market alone. The key is the delta features (model - market) which capture where the model disagrees.
- **Less is more (mostly).** Pruning 11 zero-importance features improved walk-forward ROI from +4.6% to +8.6%. However, targeted features that capture *volatility and momentum* (scoring variance, form acceleration, margin trend) did help -- the key is adding features grounded in a specific hypothesis rather than throwing data at the model.
- **Recency matters.** Exponential sample weighting (half-life 3 years) improved recent-year performance significantly by letting the model adapt to evolving league dynamics. Old seasons still contribute but don't dominate.
- **Underdogs mostly don't work.** Broad underdog betting with split edge thresholds consistently lost money. But situational filtering (home team + hot form + close scoring) identified a narrow profitable niche. The "home hot-dog" trifecta adds 6 bets at a positive ROI, improving overall performance.
- **Line betting doesn't work.** The bookmaker's margin prediction is better than our margin regressor (RMSE 33.2 vs 34.4). At ~1.91 line odds, you need >52.4% accuracy to overcome the juice. When our model disagrees with the book by >5 points, we're wrong 51.5% of the time.
- **Strategy matters more than model.** Switching from "bet anything with edge" to "favourite-only with strict filters" flipped the backtest from -$181 to +$224 -- a larger effect than any modelling improvement.
- **Quarter-Kelly is conservative enough.** The 0.25 Kelly fraction with a 5% bankroll cap kept maximum drawdown under 19%, even through losing streaks.

## 6. System Architecture

```mermaid
graph TD
    A[AFL Match CSVs<br/>2009-2024] --> D[data_ingest.py]
    B[AusportsBetting<br/>Odds + Lines] --> D
    D --> E[afl_merged.parquet]
    E --> F[features.py]
    G[Open-Meteo API<br/>Weather] --> F
    H[Squiggle API<br/>Model Consensus] --> F
    F --> J[feature_matrix.parquet<br/>45 features per match]
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
features.py            Feature engineering (Elo, Glicko-2, rolling stats, weather, etc.)
model.py               Ensemble training (LogReg + LightGBM + XGBoost + MarginReg + stacker)
backtest.py            Walk-forward backtesting engine
strategy.py            Bet selection with hot-dog filter and split edge thresholds
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
6. **v6** -- Betfair Exchange integration (live only), enhanced Squiggle features, Glicko-2 ratings, context features, neural net with team embeddings. Marginal improvement.
7. **v7** -- Systematic pruning. Removed neural net, dropped 11 zero-importance features (47 -> 36). Tested and rejected 12 new features (conversion rates, line movement, extended team stats, ladder position, ground dimensions). Less is more: ROI improved from +4.6% to +8.6%.
8. **v8** -- Rigor pass. Fixed historical enhanced-Squiggle leakage by ranking top models using only prior rounds, and tightened Squiggle joins to use round-level matching before falling back to pair-level matching. Walk-forward ROI dropped from `+8.6%` to `+6.5%`, which is less flattering but more credible.
9. **v9** -- Recency and volatility. Added exponential sample weighting (half-life 3 years) to prioritise recent seasons during training. Added three new feature families grounded in underdog upset analysis: scoring volatility (std dev), form acceleration (short vs long-term divergence), and margin trend (slope). Extensively tested underdog betting with split edge thresholds and favourite-longshot bias analysis -- confirmed dogs consistently hurt. ROI improved from `+6.5%` to `+8.8%` (68 bets, $1,203 final bankroll).
10. **v10** -- Hot-dog filter and line betting investigation. Added situational "home hot-dog" filter for selective underdog betting: home dogs with hot form (>=40% last 5) and close scoring EWMA (gap <=10) qualify at 5% edge. This adds 6 profitable dog bets. Investigated line/handicap betting using the margin regressor vs bookmaker spreads -- the book's margin RMSE (33.2) is lower than ours (34.4), so line betting is not viable. Tested odds drift features (open-to-close implied probability movement) -- hurt walk-forward performance, reverted. Ingested line data (handicap, line odds) from AusSportsBetting for future analysis. ROI improved from `+8.8%` to `+12.8%` (74 bets, $1,335 final bankroll, Sharpe 1.19).

The biggest single improvement came not from better modelling but from better strategy (v5). The second biggest came from removing features and models (v7). The third came from targeted feature additions grounded in specific hypotheses rather than data mining (v9). The most recent improvement (v10) came from narrowing down *which* underdogs to bet rather than excluding them entirely.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
