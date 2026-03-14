"""Favourite-only betting strategy with tighter thresholds."""

import pandas as pd

from config import EDGE_THRESHOLD, MAX_ODDS, MIN_MODEL_PROB
from sizing import edge, kelly_stake


class FavouriteOnlyStrategy:
    """Only bet the side where both model and market agree it's the favourite.

    Filters:
    - Model probability > MIN_MODEL_PROB (default 0.55)
    - Market implied probability > 0.5 (i.e. the market favourite)
    - Odds <= MAX_ODDS (default 3.0)
    - Edge > EDGE_THRESHOLD (default 5%)

    Returns at most one bet per match.
    """

    def __init__(
        self,
        edge_threshold: float = EDGE_THRESHOLD,
        max_odds: float = MAX_ODDS,
        min_model_prob: float = MIN_MODEL_PROB,
    ):
        self.edge_threshold = edge_threshold
        self.max_odds = max_odds
        self.min_model_prob = min_model_prob

    def select_bets(self, row: pd.Series, prob_home: float, bankroll: float) -> list:
        """Return at most one bet per match — favourite side only.

        Args:
            row: Match row with odds_home, odds_away, home_win, market_prob_home, etc.
            prob_home: Model's predicted probability of home win.
            bankroll: Current bankroll for Kelly sizing.

        Returns:
            List with zero or one bet dict.
        """
        prob_away = 1.0 - prob_home
        market_prob_home = row.get("market_prob_home", 0.5)
        market_prob_away = 1.0 - market_prob_home

        candidates = []

        # Check home side
        odds_home = row.get("odds_home")
        if odds_home is not None and not pd.isna(odds_home):
            if (
                prob_home >= self.min_model_prob
                and market_prob_home > 0.5
                and odds_home <= self.max_odds
            ):
                home_edge = edge(prob_home, odds_home)
                if home_edge > self.edge_threshold:
                    stake = kelly_stake(prob_home, odds_home, bankroll)
                    if stake > 0:
                        odds_close = row.get("odds_home_close", odds_home)
                        if pd.isna(odds_close):
                            odds_close = odds_home
                        candidates.append({
                            "side": "home",
                            "model_prob": prob_home,
                            "odds": odds_home,
                            "edge": home_edge,
                            "stake": stake,
                            "won": row["home_win"] == 1,
                            "odds_close": odds_close,
                        })

        # Check away side
        odds_away = row.get("odds_away")
        if odds_away is not None and not pd.isna(odds_away):
            if (
                prob_away >= self.min_model_prob
                and market_prob_away > 0.5
                and odds_away <= self.max_odds
            ):
                away_edge = edge(prob_away, odds_away)
                if away_edge > self.edge_threshold:
                    stake = kelly_stake(prob_away, odds_away, bankroll)
                    if stake > 0:
                        odds_close = row.get("odds_away_close", odds_away)
                        if pd.isna(odds_close):
                            odds_close = odds_away
                        candidates.append({
                            "side": "away",
                            "model_prob": prob_away,
                            "odds": odds_away,
                            "edge": away_edge,
                            "stake": stake,
                            "won": row["home_win"] == 0,
                            "odds_close": odds_close,
                        })

        if not candidates:
            return []

        # At most one bet: pick the side with the highest edge
        return [max(candidates, key=lambda b: b["edge"])]
