"""Bet tracking with SQLite."""

import sqlite3
from datetime import datetime
from config import DB_PATH


class BetTracker:
    """SQLite-backed bet tracker."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    date TEXT,
                    match TEXT,
                    side TEXT,
                    odds REAL,
                    stake REAL,
                    model_prob REAL,
                    bookmaker TEXT,
                    status TEXT DEFAULT 'open',
                    result TEXT,
                    pnl REAL,
                    settled_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    balance REAL,
                    event TEXT
                )
            """)

    def log_bet(self, date: str, match: str, side: str, odds: float,
                stake: float, model_prob: float, bookmaker: str = "") -> int:
        """Log a new bet. Returns bet ID."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO bets (date, match, side, odds, stake, model_prob, bookmaker)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (date, match, side, odds, stake, model_prob, bookmaker),
            )
            return cur.lastrowid

    def settle_bet(self, bet_id: int, won: bool):
        """Settle a bet as won or lost."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT odds, stake FROM bets WHERE id = ?", (bet_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Bet {bet_id} not found")

            odds, stake = row
            pnl = stake * (odds - 1) if won else -stake
            result = "won" if won else "lost"

            conn.execute(
                """UPDATE bets SET status = 'settled', result = ?, pnl = ?,
                   settled_at = ? WHERE id = ?""",
                (result, pnl, datetime.now().isoformat(), bet_id),
            )

            # Log bankroll change
            balance = self._get_current_balance(conn)
            balance += pnl
            conn.execute(
                "INSERT INTO bankroll_log (balance, event) VALUES (?, ?)",
                (balance, f"Settled bet #{bet_id}: {result} ({pnl:+.2f})"),
            )
            return pnl

    def _get_current_balance(self, conn) -> float:
        """Get latest bankroll balance."""
        row = conn.execute(
            "SELECT balance FROM bankroll_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else 1000.0

    def get_open_bets(self) -> list:
        """Get all open (unsettled) bets."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM bets WHERE status = 'open' ORDER BY date"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_bets(self) -> list:
        """Get all bets."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM bets ORDER BY date"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_performance_summary(self) -> dict:
        """Compute P&L summary across all settled bets."""
        with sqlite3.connect(self.db_path) as conn:
            settled = conn.execute(
                "SELECT * FROM bets WHERE status = 'settled'"
            ).fetchall()

            if not settled:
                return {"total_bets": 0, "message": "No settled bets"}

            total = len(settled)
            wins = sum(1 for r in settled if r[9] == "won")  # result column
            total_staked = sum(r[6] for r in settled)  # stake column
            total_pnl = sum(r[10] for r in settled if r[10])  # pnl column

            return {
                "total_bets": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0,
                "total_staked": total_staked,
                "total_pnl": total_pnl,
                "roi": total_pnl / total_staked if total_staked > 0 else 0,
            }

    def get_monthly_breakdown(self) -> list:
        """Get P&L breakdown by month."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT
                    strftime('%Y-%m', date) as month,
                    COUNT(*) as bets,
                    SUM(CASE WHEN result = 'won' THEN 1 ELSE 0 END) as wins,
                    SUM(stake) as staked,
                    SUM(COALESCE(pnl, 0)) as pnl
                FROM bets
                WHERE status = 'settled'
                GROUP BY month
                ORDER BY month
            """).fetchall()
            return [
                {
                    "month": r[0],
                    "bets": r[1],
                    "wins": r[2],
                    "staked": r[3],
                    "pnl": r[4],
                    "roi": r[4] / r[3] if r[3] else 0,
                }
                for r in rows
            ]

    def init_bankroll(self, amount: float):
        """Set initial bankroll balance."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO bankroll_log (balance, event) VALUES (?, ?)",
                (amount, "Initial bankroll"),
            )
