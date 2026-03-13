"""Kelly criterion bet sizing."""

from config import KELLY_FRACTION, MAX_BET_FRACTION, MIN_STAKE


def kelly_stake(prob: float, odds: float, bankroll: float,
                fraction: float = KELLY_FRACTION,
                max_frac: float = MAX_BET_FRACTION,
                min_stake: float = MIN_STAKE) -> float:
    """Compute Kelly criterion stake.

    Args:
        prob: Model-estimated probability of winning.
        odds: Decimal odds offered by bookmaker.
        bankroll: Current bankroll.
        fraction: Kelly fraction (e.g., 0.25 for quarter-Kelly).
        max_frac: Maximum fraction of bankroll per bet.
        min_stake: Minimum stake (skip if below).

    Returns:
        Recommended stake (0 if no edge or below minimum).
    """
    # Kelly formula: f* = (bp - q) / b
    # where b = odds - 1, p = prob, q = 1 - p
    b = odds - 1
    if b <= 0:
        return 0.0

    q = 1 - prob
    kelly_frac = (b * prob - q) / b

    if kelly_frac <= 0:
        return 0.0

    # Apply fractional Kelly
    stake_frac = kelly_frac * fraction

    # Cap at max fraction
    stake_frac = min(stake_frac, max_frac)

    stake = stake_frac * bankroll

    # Enforce minimum
    if stake < min_stake:
        return 0.0

    return round(stake, 2)


def edge(prob: float, odds: float) -> float:
    """Compute edge: model_prob * odds - 1 (EV per dollar)."""
    return prob * odds - 1
