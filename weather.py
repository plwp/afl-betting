"""Fetch and cache historical weather data from Open-Meteo API."""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from config import VENUE_COORDS, ROOFED_VENUES, WEATHER_CACHE_DIR


def _cache_path(venue: str, date_str: str) -> str:
    os.makedirs(WEATHER_CACHE_DIR, exist_ok=True)
    safe_venue = venue.replace("/", "_").replace(" ", "_").replace(".", "")
    return os.path.join(WEATHER_CACHE_DIR, f"{safe_venue}_{date_str}.json")


def fetch_weather(venue: str, match_date: str) -> dict:
    """Fetch weather for a venue on a given date. Returns rain_mm and wind_speed.

    Args:
        venue: Venue name (must be in VENUE_COORDS).
        match_date: Date string in YYYY-MM-DD format.

    Returns:
        dict with keys: rain_mm, wind_speed, wind_gusts
    """
    # Roofed venues are weather-neutral
    if venue in ROOFED_VENUES:
        return {"rain_mm": 0.0, "wind_speed": 0.0, "wind_gusts": 0.0}

    coords = VENUE_COORDS.get(venue)
    if coords is None:
        return {"rain_mm": 0.0, "wind_speed": 0.0, "wind_gusts": 0.0}

    # Check cache
    cache_file = _cache_path(venue, match_date)
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)

    lat, lon = coords
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={match_date}&end_date={match_date}"
        f"&hourly=rain,wind_speed_10m,wind_gusts_10m"
        f"&timezone=Australia%2FSydney"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        rain_values = hourly.get("rain", [])
        wind_values = hourly.get("wind_speed_10m", [])
        gust_values = hourly.get("wind_gusts_10m", [])

        # AFL games are typically 12:00-20:00 local time. Use hours 12-20.
        game_hours = slice(12, 21)
        rain_mm = float(np.sum(rain_values[game_hours])) if rain_values else 0.0
        wind_speed = float(np.mean(wind_values[game_hours])) if wind_values else 0.0
        wind_gusts = float(np.max(gust_values[game_hours])) if gust_values else 0.0

        result = {
            "rain_mm": round(rain_mm, 1),
            "wind_speed": round(wind_speed, 1),
            "wind_gusts": round(wind_gusts, 1),
        }
    except Exception:
        result = {"rain_mm": 0.0, "wind_speed": 0.0, "wind_gusts": 0.0}

    # Cache
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def fetch_weather_batch(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather for all matches. Adds rain_mm, wind_speed, is_wet, is_roofed columns.

    Expects columns: venue, date.
    Rate-limits requests to respect Open-Meteo fair use.
    """
    results = []
    total = len(matches_df)
    api_calls = 0

    for i, row in matches_df.iterrows():
        venue = row["venue"]
        date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

        # Check if cached before counting as API call
        cache_file = _cache_path(venue, date_str)
        needs_api = not os.path.exists(cache_file) and venue not in ROOFED_VENUES and venue in VENUE_COORDS

        weather = fetch_weather(venue, date_str)

        if needs_api:
            api_calls += 1
            # Rate limit: max 10 requests per second for Open-Meteo
            if api_calls % 10 == 0:
                time.sleep(1.1)
            if api_calls % 100 == 0:
                print(f"  Weather: {api_calls} API calls ({i+1}/{total} matches)")

        results.append({
            "rain_mm": weather["rain_mm"],
            "wind_speed": weather["wind_speed"],
            "is_wet": int(weather["rain_mm"] > 1.0),
            "is_roofed": int(venue in ROOFED_VENUES),
        })

    weather_df = pd.DataFrame(results, index=matches_df.index)
    if api_calls > 0:
        print(f"  Weather: {api_calls} API calls total")
    return weather_df
