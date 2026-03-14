"""Betfair Exchange API client — back/lay spread and matched volume."""

import os
import json
import time
import hashlib
import requests

from config import TEAM_NAME_MAP

BETFAIR_LOGIN_URL = "https://identitysso.betfair.com.au/api/login"
BETFAIR_API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0/"
BETFAIR_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".betfair_cache")
BETFAIR_CACHE_TTL = 900  # 15 minutes

# Betfair competition ID for AFL
AFL_COMPETITION_ID = "11897406"

# Betfair team name mapping to canonical names
BETFAIR_TEAM_MAP = {
    "Brisbane Lions": "Brisbane",
    "Greater Western Sydney": "GWS Giants",
    "GWS Giants": "GWS Giants",
    "GWS GIANTS": "GWS Giants",
    "Gold Coast Suns": "Gold Coast",
    "Adelaide Crows": "Adelaide",
    "West Coast Eagles": "West Coast",
    "Sydney Swans": "Sydney",
    "Western Bulldogs": "Western Bulldogs",
    "North Melbourne": "North Melbourne",
    "St Kilda": "St Kilda",
    "Port Adelaide": "Port Adelaide",
    "Geelong Cats": "Geelong",
}


def _normalize_betfair_team(name: str) -> str:
    """Normalize Betfair runner names to canonical form."""
    if name in BETFAIR_TEAM_MAP:
        return BETFAIR_TEAM_MAP[name]
    return TEAM_NAME_MAP.get(name, name)


def _cache_path(key_str: str) -> str:
    os.makedirs(BETFAIR_CACHE_DIR, exist_ok=True)
    h = hashlib.md5(key_str.encode()).hexdigest()
    return os.path.join(BETFAIR_CACHE_DIR, f"{h}.json")


def _read_cache(key_str: str):
    path = _cache_path(key_str)
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        if time.time() - mtime < BETFAIR_CACHE_TTL:
            with open(path) as f:
                return json.load(f)
    return None


def _write_cache(key_str: str, data):
    path = _cache_path(key_str)
    with open(path, "w") as f:
        json.dump({"timestamp": time.time(), "data": data}, f)


class BetfairClient:
    """Betfair Exchange client using non-interactive SSO login."""

    def __init__(self):
        self.username = os.getenv("BETFAIR_USERNAME", "")
        self.password = os.getenv("BETFAIR_PASSWORD", "")
        self.app_key = os.getenv("BETFAIR_APP_KEY", "")
        self.session_token = None

    def _is_configured(self) -> bool:
        return bool(self.username and self.password and self.app_key)

    def login(self):
        """Authenticate via Betfair SSO. Returns True on success."""
        if not self._is_configured():
            return False

        resp = requests.post(
            BETFAIR_LOGIN_URL,
            data={"username": self.username, "password": self.password},
            headers={
                "X-Application": self.app_key,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") == "SUCCESS":
            self.session_token = result["token"]
            return True
        return False

    def _api_headers(self) -> dict:
        return {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _api_call(self, method: str, params: dict) -> dict:
        """Make a Betfair API call."""
        url = BETFAIR_API_URL + method + "/"
        resp = requests.post(
            url,
            json=params,
            headers=self._api_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def list_afl_markets(self) -> list:
        """Find AFL Match Odds markets via listMarketCatalogue."""
        cache_key = "afl_markets"
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached["data"]

        params = {
            "filter": {
                "competitionIds": [AFL_COMPETITION_ID],
                "marketTypeCodes": ["MATCH_ODDS"],
            },
            "maxResults": "100",
            "marketProjection": [
                "RUNNER_DESCRIPTION",
                "EVENT",
                "COMPETITION",
                "MARKET_START_TIME",
            ],
        }

        markets = self._api_call("listMarketCatalogue", params)
        _write_cache(cache_key, markets)
        return markets

    def get_market_data(self, market_id: str) -> dict:
        """Get back/lay prices and traded volume for a market."""
        cache_key = f"market_{market_id}"
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached["data"]

        params = {
            "marketIds": [market_id],
            "priceProjection": {
                "priceData": ["EX_BEST_OFFERS"],
            },
            "currencyCode": "AUD",
        }

        books = self._api_call("listMarketBook", params)
        if not books:
            return {}

        book = books[0]
        _write_cache(cache_key, book)
        return book

    def get_afl_markets(self) -> list:
        """Return enriched AFL market data with spreads and volume."""
        markets = self.list_afl_markets()
        results = []

        for market in markets:
            market_id = market.get("marketId", "")
            event = market.get("event", {})
            runners = market.get("runners", [])

            book = self.get_market_data(market_id)
            if not book:
                continue

            book_runners = {r["selectionId"]: r for r in book.get("runners", [])}

            selections = []
            for runner in runners:
                sel_id = runner.get("selectionId")
                name = _normalize_betfair_team(runner.get("runnerName", ""))
                book_runner = book_runners.get(sel_id, {})

                ex = book_runner.get("ex", {})
                back_prices = ex.get("availableToBack", [])
                lay_prices = ex.get("availableToLay", [])

                best_back = back_prices[0]["price"] if back_prices else 0
                best_lay = lay_prices[0]["price"] if lay_prices else 0
                spread = (best_lay - best_back) if (best_back > 0 and best_lay > 0) else 0
                # Use available back volume as proxy (traded volume needs non-delayed key)
                back_volume = sum(p.get("size", 0) for p in back_prices)
                total_matched = back_volume

                selections.append({
                    "name": name,
                    "best_back": best_back,
                    "best_lay": best_lay,
                    "spread": spread,
                    "matched_volume": total_matched,
                })

            total_volume = sum(s["matched_volume"] for s in selections)

            results.append({
                "market_id": market_id,
                "event_name": event.get("name", ""),
                "start_time": market.get("marketStartTime", ""),
                "selections": selections,
                "total_matched": book.get("totalMatched", 0),
                "total_volume": total_volume,
            })

            time.sleep(0.2)  # Rate limit

        return results


def get_betfair_data() -> dict:
    """Fetch Betfair data and return a dict keyed by (home_team, away_team).

    Each value contains: bf_spread_home, bf_spread_away, bf_volume_ratio.
    Returns empty dict if Betfair is not configured or login fails.
    """
    client = BetfairClient()
    if not client._is_configured():
        return {}

    try:
        if not client.login():
            print("Betfair login failed — skipping exchange data")
            return {}
    except Exception as e:
        print(f"Betfair login error: {e}")
        return {}

    try:
        markets = client.get_afl_markets()
    except Exception as e:
        print(f"Betfair API error: {e}")
        return {}

    result = {}
    for market in markets:
        sels = market.get("selections", [])
        if len(sels) < 2:
            continue

        # Try to identify home/away from event name (typically "Team A v Team B")
        event_name = market.get("event_name", "")
        parts = event_name.split(" v ")
        if len(parts) != 2:
            parts = event_name.split(" vs ")
        if len(parts) != 2:
            continue

        home = _normalize_betfair_team(parts[0].strip())
        away = _normalize_betfair_team(parts[1].strip())

        # Match selections to home/away
        home_sel = next((s for s in sels if s["name"] == home), None)
        away_sel = next((s for s in sels if s["name"] == away), None)
        if not home_sel or not away_sel:
            # Fallback: first two selections
            home_sel, away_sel = sels[0], sels[1]

        total_vol = home_sel["matched_volume"] + away_sel["matched_volume"]
        vol_ratio = (
            home_sel["matched_volume"] / total_vol if total_vol > 0 else 0.5
        )

        result[(home, away)] = {
            "bf_spread_home": home_sel["spread"],
            "bf_spread_away": away_sel["spread"],
            "bf_volume_ratio": vol_ratio,
        }

    if result:
        print(f"Betfair: {len(result)} markets loaded")

    return result
