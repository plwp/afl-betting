"""Betting strategy with split edge thresholds, situational dog filter, and line betting."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import EDGE_THRESHOLD, EDGE_THRESHOLD_DOG, MAX_ODDS, MIN_MODEL_PROB
from sizing import edge, kelly_stake

# Hot-dog thresholds: home dog with hot form and close scoring
DOG_MIN_FORM = 0.4          # won 2+ of last 5
DOG_MAX_SCORING_GAP = 10.0  # scoring EWMA within 10 points of opponent


class BettingStrategy:
    """Bet favourites on edge, underdogs only when situational signals align.

    Favourites: standard edge > threshold.
    Underdogs: must be home team, hot recent form, and close on scoring --
    the "home hot-dog" trifecta that showed +10% ROI historically.
    """

    def __init__(
        self,
        edge_threshold_fav: float = EDGE_THRESHOLD,
        edge_threshold_dog: float = EDGE_THRESHOLD_DOG,
        max_odds: float = MAX_ODDS,
        min_model_prob: float = MIN_MODEL_PROB,
    ):
        self.edge_threshold_fav = edge_threshold_fav
        self.edge_threshold_dog = edge_threshold_dog
        self.max_odds = max_odds
        self.min_model_prob = min_model_prob

    def _is_hot_dog(self, row, side) -> bool:
        """Check if this underdog qualifies as a 'home hot-dog' bet."""
        if side == "home":
            form = row.get("form_home_5", 0)
            scoring = row.get("scoring_ewma_home", 0)
            scoring_opp = row.get("scoring_ewma_away", 0)
        else:
            form = row.get("form_away_5", 0)
            scoring = row.get("scoring_ewma_away", 0)
            scoring_opp = row.get("scoring_ewma_home", 0)
            # Must be home team — away dogs don't qualify
            return False

        scoring_gap = scoring_opp - scoring  # positive = opponent outscoring
        return form >= DOG_MIN_FORM and scoring_gap <= DOG_MAX_SCORING_GAP

    def _check_side(self, row, side, model_prob, market_prob, odds_key, win_cond, bankroll):
        odds = row.get(odds_key)
        if odds is None or pd.isna(odds):
            return None
        if model_prob < self.min_model_prob or odds > self.max_odds:
            return None

        is_fav = market_prob > 0.5

        if is_fav:
            threshold = self.edge_threshold_fav
        else:
            # Dogs must pass the hot-dog situational filter
            if not self._is_hot_dog(row, side):
                return None
            threshold = self.edge_threshold_dog

        e = edge(model_prob, odds)
        if e <= threshold:
            return None

        stake = kelly_stake(model_prob, odds, bankroll)
        if stake <= 0:
            return None

        if side == "home":
            odds_close = row.get("odds_home_close", odds)
        else:
            odds_close = row.get("odds_away_close", odds)
        if pd.isna(odds_close):
            odds_close = odds

        return {
            "side": side,
            "model_prob": model_prob,
            "odds": odds,
            "edge": e,
            "stake": stake,
            "won": win_cond,
            "odds_close": odds_close,
        }

    def _check_line_side(self, row, side, margin_pred, margin_std, bankroll):
        """Evaluate a line bet using margin prediction vs bookmaker's line."""
        line = row.get("home_line_close")
        if line is None or pd.isna(line):
            return None

        if side == "home":
            line_odds = row.get("home_line_odds_close")
            # Home covers when margin + line > 0
            cover_prob = float(norm.cdf((margin_pred + line) / margin_std))
        else:
            line_odds = row.get("away_line_odds_close")
            # Away covers when margin + line < 0
            cover_prob = 1.0 - float(norm.cdf((margin_pred + line) / margin_std))

        if line_odds is None or pd.isna(line_odds) or line_odds <= 1.0:
            return None

        e = edge(cover_prob, line_odds)
        if e <= self.edge_threshold_fav:
            return None

        stake = kelly_stake(cover_prob, line_odds, bankroll)
        if stake <= 0:
            return None

        # Determine if the line bet actually won
        actual_margin = row.get("margin", 0)
        if side == "home":
            won = (actual_margin + line) > 0
        else:
            won = (actual_margin + line) < 0

        return {
            "side": f"line_{side}",
            "model_prob": cover_prob,
            "odds": line_odds,
            "edge": e,
            "stake": stake,
            "won": bool(won),
            "odds_close": line_odds,  # line odds don't have separate open/close we track
        }

    def select_bets(
        self, row: pd.Series, prob_home: float, bankroll: float,
        margin_pred: float = None, margin_std: float = None,
    ) -> list:
        market_prob_home = row.get("market_prob_home", 0.5)

        candidates = []

        # Head-to-head bets
        home = self._check_side(
            row, "home", prob_home, market_prob_home,
            "odds_home", row["home_win"] == 1, bankroll,
        )
        if home:
            candidates.append(home)

        away = self._check_side(
            row, "away", 1.0 - prob_home, 1.0 - market_prob_home,
            "odds_away", row["home_win"] == 0, bankroll,
        )
        if away:
            candidates.append(away)

        # Line bets disabled: margin model RMSE (34.4) > book (33.2),
        # disagreements lose money. Keep infrastructure for future use.
        # if margin_pred is not None and margin_std is not None and margin_std > 0:
        #     line_home = self._check_line_side(row, "home", margin_pred, margin_std, bankroll)
        #     if line_home:
        #         candidates.append(line_home)
        #     line_away = self._check_line_side(row, "away", margin_pred, margin_std, bankroll)
        #     if line_away:
        #         candidates.append(line_away)

        if not candidates:
            return []
        return [max(candidates, key=lambda b: b["edge"])]


# Backward-compatible alias
FavouriteOnlyStrategy = BettingStrategy
