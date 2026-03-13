#!/usr/bin/env python3
"""CLI performance report for bet tracking."""

import argparse
from tabulate import tabulate
from tracker import BetTracker
from config import INITIAL_BANKROLL


def main():
    parser = argparse.ArgumentParser(description="AFL Betting Performance Report")
    parser.add_argument("--settle", nargs=2, metavar=("BET_ID", "WON"),
                        help="Settle a bet: --settle <id> <won|lost>")
    parser.add_argument("--open", action="store_true",
                        help="Show open bets")
    parser.add_argument("--init-bankroll", type=float,
                        help="Initialize bankroll balance")
    args = parser.parse_args()

    tracker = BetTracker()

    # Handle settle
    if args.settle:
        bet_id = int(args.settle[0])
        won = args.settle[1].lower() in ("won", "true", "1", "yes", "w")
        pnl = tracker.settle_bet(bet_id, won)
        print(f"Bet #{bet_id} settled: {'WON' if won else 'LOST'}, P&L: ${pnl:+.2f}")
        return

    # Handle init bankroll
    if args.init_bankroll:
        tracker.init_bankroll(args.init_bankroll)
        print(f"Bankroll initialized at ${args.init_bankroll:.2f}")
        return

    # Show open bets
    if args.open:
        open_bets = tracker.get_open_bets()
        if not open_bets:
            print("No open bets.")
            return
        print(f"\n{'='*60}")
        print(f"OPEN BETS: {len(open_bets)}")
        print(f"{'='*60}")
        headers = ["ID", "Date", "Match", "Side", "Odds", "Stake", "Bookmaker"]
        rows = [
            [b["id"], b["date"][:10] if b["date"] else "", b["match"],
             b["side"], f"{b['odds']:.2f}", f"${b['stake']:.2f}", b["bookmaker"]]
            for b in open_bets
        ]
        print(tabulate(rows, headers=headers, tablefmt="simple"))
        return

    # Full performance report
    summary = tracker.get_performance_summary()

    print(f"\n{'='*60}")
    print("AFL BETTING PERFORMANCE REPORT")
    print(f"{'='*60}")

    if summary["total_bets"] == 0:
        print("\nNo settled bets yet.")
        open_bets = tracker.get_open_bets()
        if open_bets:
            print(f"{len(open_bets)} open bets pending.")
        return

    print(f"\n  Total Bets:    {summary['total_bets']}")
    print(f"  Wins/Losses:   {summary['wins']}/{summary['losses']}")
    print(f"  Win Rate:      {summary['win_rate']:.1%}")
    print(f"  Total Staked:  ${summary['total_staked']:,.2f}")
    print(f"  Total P&L:     ${summary['total_pnl']:+,.2f}")
    print(f"  ROI:           {summary['roi']:+.1%}")

    # Monthly breakdown
    monthly = tracker.get_monthly_breakdown()
    if monthly:
        print(f"\n{'='*60}")
        print("MONTHLY BREAKDOWN")
        print(f"{'='*60}")
        headers = ["Month", "Bets", "Wins", "Staked", "P&L", "ROI"]
        rows = [
            [m["month"], m["bets"], m["wins"],
             f"${m['staked']:,.2f}", f"${m['pnl']:+,.2f}", f"{m['roi']:+.1%}"]
            for m in monthly
        ]
        print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Open bets
    open_bets = tracker.get_open_bets()
    if open_bets:
        total_exposure = sum(b["stake"] for b in open_bets)
        print(f"\n  Open Bets:     {len(open_bets)} (${total_exposure:,.2f} exposure)")


if __name__ == "__main__":
    main()
