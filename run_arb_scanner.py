#!/usr/bin/env python3
"""Dedicated cross-bookie arbitrage scanner. No model needed."""

import argparse
import sys
from datetime import datetime

import pandas as pd
from tabulate import tabulate

from config import (
    ARB_STAKE_FRACTION, AU_BOOKMAKERS, INITIAL_BANKROLL, MIN_STAKE,
    ODDS_API_BASE, ODDS_API_KEY,
)
from scanner import fetch_odds, _normalize_api_team


def parse_all_odds(events: list, au_only: bool = False) -> list[dict]:
    """Parse API response keeping every bookie's price per side.

    Returns one dict per event with:
      - home_team, away_team, commence_time
      - home_odds: {bookie: price, ...}
      - away_odds: {bookie: price, ...}
      - best_home_odds, best_home_bookie
      - best_away_odds, best_away_bookie
    """
    parsed = []
    for event in events:
        home = _normalize_api_team(event.get("home_team", ""))
        away = _normalize_api_team(event.get("away_team", ""))
        commence = event.get("commence_time", "")

        home_odds = {}
        away_odds = {}

        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            if au_only and AU_BOOKMAKERS and bm_key not in AU_BOOKMAKERS:
                continue
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = _normalize_api_team(outcome.get("name", ""))
                    price = outcome.get("price", 0)
                    if price <= 1:
                        continue
                    if team == home:
                        home_odds[bm_key] = price
                    elif team == away:
                        away_odds[bm_key] = price

        best_home = max(home_odds.values()) if home_odds else 0
        best_away = max(away_odds.values()) if away_odds else 0
        best_home_bk = max(home_odds, key=home_odds.get) if home_odds else ""
        best_away_bk = max(away_odds, key=away_odds.get) if away_odds else ""

        parsed.append({
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "home_odds": home_odds,
            "away_odds": away_odds,
            "best_home_odds": best_home,
            "best_home_bookie": best_home_bk,
            "best_away_odds": best_away,
            "best_away_bookie": best_away_bk,
        })

    return parsed


def find_arbs(matches: list[dict], bankroll: float,
              stake_fraction: float = ARB_STAKE_FRACTION,
              near_threshold: float = 0.02) -> tuple[list, list]:
    """Find arbitrage and near-arbitrage opportunities.

    Args:
        matches: Output of parse_all_odds().
        bankroll: Current bankroll for stake sizing.
        stake_fraction: Fraction of bankroll to deploy per arb.
        near_threshold: Show near-arbs where overround < this (e.g. 0.02 = 2%).

    Returns:
        (arbs, near_arbs) — each a list of dicts.
    """
    arbs = []
    near_arbs = []

    for m in matches:
        best_h = m["best_home_odds"]
        best_a = m["best_away_odds"]
        if best_h <= 1 or best_a <= 1:
            continue

        inv_sum = 1 / best_h + 1 / best_a
        overround = inv_sum - 1  # negative = arb, positive = bookie margin

        match_label = f"{m['home_team']} v {m['away_team']}"

        if overround < 0:
            # True arb
            total = bankroll * stake_fraction
            stake_h = round(total * (1 / best_h) / inv_sum, 2)
            stake_a = round(total * (1 / best_a) / inv_sum, 2)
            profit = round(total / inv_sum - total, 2)

            if stake_h < MIN_STAKE or stake_a < MIN_STAKE:
                continue

            arbs.append({
                "match": match_label,
                "commence": m["commence_time"],
                "home_odds": best_h,
                "home_bookie": m["best_home_bookie"],
                "away_odds": best_a,
                "away_bookie": m["best_away_bookie"],
                "overround": overround,
                "stake_home": stake_h,
                "stake_away": stake_a,
                "total_stake": round(stake_h + stake_a, 2),
                "profit": profit,
                "roi": profit / (stake_h + stake_a),
            })
        elif overround < near_threshold:
            # Near-arb — worth watching
            near_arbs.append({
                "match": match_label,
                "commence": m["commence_time"],
                "home_odds": best_h,
                "home_bookie": m["best_home_bookie"],
                "away_odds": best_a,
                "away_bookie": m["best_away_bookie"],
                "overround": overround,
            })

    arbs.sort(key=lambda a: a["overround"])
    near_arbs.sort(key=lambda a: a["overround"])
    return arbs, near_arbs


def main():
    parser = argparse.ArgumentParser(description="AFL Cross-Bookie Arbitrage Scanner")
    parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL,
                        help="Current bankroll")
    parser.add_argument("--stake-pct", type=float, default=ARB_STAKE_FRACTION * 100,
                        help="Percent of bankroll per arb (default: 5)")
    parser.add_argument("--near", type=float, default=2.0,
                        help="Show near-arbs with overround below this %% (default: 2)")
    parser.add_argument("--all-bookies", action="store_true",
                        help="Include international bookmakers (not just AU)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force refresh odds (ignore cache)")
    args = parser.parse_args()

    # Fetch odds
    try:
        events = fetch_odds(force_refresh=args.refresh)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"API error: {e}")
        return

    if not events:
        print("No upcoming AFL events found.")
        return

    # Parse all bookie prices
    matches = parse_all_odds(events, au_only=not args.all_bookies)
    n_bookies = set()
    for m in matches:
        n_bookies.update(m["home_odds"].keys())
        n_bookies.update(m["away_odds"].keys())

    scope = "all" if args.all_bookies else "AU"
    print(f"\n{len(matches)} matches, {len(n_bookies)} {scope} bookmakers")

    # Show all odds for each match
    print(f"\n{'='*70}")
    print("ODDS MATRIX")
    print(f"{'='*70}")
    for m in matches:
        print(f"\n  {m['home_team']} v {m['away_team']}")
        all_bookies = sorted(set(m["home_odds"].keys()) | set(m["away_odds"].keys()))
        for bk in all_bookies:
            h = m["home_odds"].get(bk, "-")
            a = m["away_odds"].get(bk, "-")
            h_str = f"{h:.2f}" if isinstance(h, (int, float)) else h
            a_str = f"{a:.2f}" if isinstance(a, (int, float)) else a
            print(f"    {bk:20s}  {h_str:>6s} / {a_str:<6s}")
        inv = 1 / m["best_home_odds"] + 1 / m["best_away_odds"] if m["best_home_odds"] > 0 and m["best_away_odds"] > 0 else 0
        overround = (inv - 1) * 100
        print(f"    {'BEST':20s}  {m['best_home_odds']:>6.2f} / {m['best_away_odds']:<6.2f}  (overround: {overround:+.2f}%)")

    # Find arbs
    stake_frac = args.stake_pct / 100
    arbs, near_arbs = find_arbs(matches, args.bankroll,
                                 stake_fraction=stake_frac,
                                 near_threshold=args.near / 100)

    if arbs:
        total_profit = sum(a["profit"] for a in arbs)
        print(f"\n{'='*70}")
        print(f"ARBITRAGE FOUND: {len(arbs)} opportunity(ies), ${total_profit:.2f} guaranteed profit")
        print(f"{'='*70}")
        for a in arbs:
            print(f"\n  {a['match']}")
            print(f"    Bet {a['home_bookie']:15s}  HOME @ {a['home_odds']:.2f}  stake ${a['stake_home']:.2f}")
            print(f"    Bet {a['away_bookie']:15s}  AWAY @ {a['away_odds']:.2f}  stake ${a['stake_away']:.2f}")
            print(f"    Total: ${a['total_stake']:.2f} -> Profit: ${a['profit']:.2f} ({a['roi']:+.2%})")
    else:
        print(f"\n{'='*70}")
        print("NO ARBITRAGE FOUND")
        print(f"{'='*70}")

    if near_arbs:
        print(f"\n{'='*70}")
        print(f"NEAR-ARBS ({len(near_arbs)} matches with overround < {args.near:.1f}%)")
        print(f"{'='*70}")
        rows = []
        for na in near_arbs:
            rows.append({
                "match": na["match"],
                "home": f"{na['home_odds']:.2f} ({na['home_bookie']})",
                "away": f"{na['away_odds']:.2f} ({na['away_bookie']})",
                "overround": f"{na['overround']:+.2%}",
            })
        print(tabulate(rows, headers="keys", tablefmt="simple", showindex=False))
        print("\n  Tip: these may flip to arbs as lines move. Re-run with --refresh to check.")

    if not arbs and not near_arbs:
        print(f"\n  No near-arbs either (all overrounds > {args.near:.1f}%).")
        print("  Try --all-bookies to include international bookmakers.")


if __name__ == "__main__":
    main()
