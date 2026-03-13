"""Configuration constants for AFL betting system."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Data Sources ---
MATCH_CSV_URL = (
    "https://raw.githubusercontent.com/akareen/AFL-Data-Analysis"
    "/main/data/matches/matches_{year}.csv"
)
ODDS_XLSX_URL = "https://www.aussportsbetting.com/historical_data/afl.xlsx"
MATCH_YEARS = range(2009, 2025)

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MERGED_PATH = os.path.join(DATA_DIR, "afl_merged.parquet")
FEATURE_PATH = os.path.join(DATA_DIR, "feature_matrix.parquet")
ODDS_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".odds_cache")

# --- Team Name Mappings ---
TEAM_NAME_MAP = {
    "Brisbane Lions": "Brisbane",
    "Greater Western Sydney": "GWS Giants",
    "Adelaide": "Adelaide",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
    "GWS Giants": "GWS Giants",
    "Brisbane": "Brisbane",
}

ALL_TEAMS = sorted(set(TEAM_NAME_MAP.values()))

# --- Elo Parameters ---
ELO_K = 30
ELO_HOME_ADV = 30
ELO_INIT = 1500
ELO_SEASON_REVERT = 0.33
ELO_MARGIN_K_CAP = 2.5  # cap on margin-based K multiplier

# --- Temporal Splits ---
TRAIN_END = 2020
VAL_START = 2021
VAL_END = 2022
TEST_START = 2023
TEST_END = 2024

# --- Betting Parameters (conservative from Gemini) ---
KELLY_FRACTION = 0.15
MAX_BET_FRACTION = 0.03
MIN_STAKE = 5.0
EDGE_THRESHOLD = 0.05
INITIAL_BANKROLL = 1000.0

# --- Team-to-State Mappings ---
TEAM_STATE = {
    "Adelaide": "SA", "Port Adelaide": "SA",
    "Brisbane": "QLD", "Gold Coast": "QLD",
    "Fremantle": "WA", "West Coast": "WA",
    "Sydney": "NSW", "GWS Giants": "NSW",
    "Carlton": "VIC", "Collingwood": "VIC", "Essendon": "VIC",
    "Geelong": "VIC", "Hawthorn": "VIC", "Melbourne": "VIC",
    "North Melbourne": "VIC", "Richmond": "VIC", "St Kilda": "VIC",
    "Western Bulldogs": "VIC",
}

VENUE_STATE = {
    "M.C.G.": "VIC", "Docklands": "VIC", "Marvel Stadium": "VIC",
    "Kardinia Park": "VIC", "GMHBA Stadium": "VIC", "Eureka Stadium": "VIC",
    "S.C.G.": "NSW", "Stadium Australia": "NSW", "Accor Stadium": "NSW",
    "Sydney Showground": "NSW", "ENGIE Stadium": "NSW", "Giants Stadium": "NSW",
    "Subiaco": "WA", "Perth Stadium": "WA", "Optus Stadium": "WA",
    "Domain Stadium": "WA",
    "Football Park": "SA", "Adelaide Oval": "SA",
    "Gabba": "QLD", "Carrara": "QLD", "People First Stadium": "QLD",
    "Cazaly's Stadium": "QLD", "Riverway Stadium": "QLD",
    "York Park": "TAS", "Bellerive Oval": "TAS", "UTAS Stadium": "TAS",
    "Blundstone Arena": "TAS",
    "Manuka Oval": "ACT",
    "Marrara Oval": "NT", "TIO Stadium": "NT", "Traeger Park": "NT",
}

# Approximate flight hours between states (symmetric)
TRAVEL_HOURS = {
    ("VIC", "SA"): 1.0, ("VIC", "QLD"): 2.5, ("VIC", "WA"): 4.0,
    ("VIC", "NSW"): 1.5, ("VIC", "TAS"): 1.0, ("VIC", "ACT"): 1.0,
    ("VIC", "NT"): 4.0,
    ("SA", "QLD"): 2.5, ("SA", "WA"): 3.0, ("SA", "NSW"): 2.0,
    ("SA", "TAS"): 2.0, ("SA", "ACT"): 2.0, ("SA", "NT"): 3.0,
    ("QLD", "WA"): 5.0, ("QLD", "NSW"): 1.5, ("QLD", "TAS"): 3.0,
    ("QLD", "ACT"): 2.0, ("QLD", "NT"): 3.0,
    ("WA", "NSW"): 4.5, ("WA", "TAS"): 5.0, ("WA", "ACT"): 4.5,
    ("WA", "NT"): 3.5,
    ("NSW", "TAS"): 2.0, ("NSW", "ACT"): 0.5, ("NSW", "NT"): 4.0,
    ("TAS", "ACT"): 2.0, ("TAS", "NT"): 4.5,
    ("ACT", "NT"): 4.0,
}

# --- Feature Columns ---
FEATURE_COLS = [
    "elo_diff", "elo_prob",
    "market_prob_home", "market_prob_away", "market_overround",
    "market_elo_delta",
    "is_home_state", "travel_hours_home", "travel_hours_away",
    "form_home_5", "form_away_5", "form_diff",
    "win_pct_home_10", "win_pct_away_10", "win_pct_diff",
    "venue_exp_home", "venue_exp_away", "venue_exp_diff",
    "rest_days_home", "rest_days_away", "rest_diff",
    "h2h_home_win_pct",
    "season_round", "is_final",
    "margin_ewma_home", "margin_ewma_away", "margin_ewma_diff",
    "scoring_ewma_home", "scoring_ewma_away", "scoring_ewma_diff",
]

# --- Odds API ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds/"
ODDS_CACHE_TTL = 900
AU_BOOKMAKERS = [
    "sportsbet", "tab", "pointsbetau", "unibet", "ladbrokes_au",
    "betfair_ex_au", "neds", "bluebet",
]

# --- Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), "bets.db")
