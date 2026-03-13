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

# --- Team Name Mappings (canonical → variants) ---
# Canonical names use the odds dataset convention.
TEAM_NAME_MAP = {
    # match CSV name  →  canonical (odds) name
    "Brisbane Lions": "Brisbane",
    "Greater Western Sydney": "GWS Giants",
    # these are already identical, but listed for completeness
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
ELO_SEASON_REVERT = 0.33  # revert 1/3 towards mean each season
ELO_WARMUP_BEFORE = 2009  # use data from 2000+ if available for warmup

# --- Temporal Splits ---
TRAIN_END = 2020      # train: <= 2020
VAL_START = 2021      # validation: 2021-2022
VAL_END = 2022
TEST_START = 2023     # test: 2023-2024
TEST_END = 2024

# --- Betting Parameters ---
KELLY_FRACTION = 0.25
MAX_BET_FRACTION = 0.05  # max 5% of bankroll
MIN_STAKE = 5.0
EDGE_THRESHOLD = 0.03    # minimum 3% edge to bet
INITIAL_BANKROLL = 1000.0

# --- Feature Columns ---
FEATURE_COLS = [
    "elo_diff", "elo_prob",
    "form_home_5", "form_away_5", "form_diff",
    "win_pct_home_10", "win_pct_away_10",
    "win_pct_diff",
    "venue_exp_home", "venue_exp_away",
    "venue_exp_diff",
    "rest_days_home", "rest_days_away", "rest_diff",
    "h2h_home_win_pct",
    "season_round",
    "margin_ewma_home", "margin_ewma_away",
    "margin_ewma_diff",
    "scoring_ewma_home", "scoring_ewma_away",
    "scoring_ewma_diff",
    "market_prob_home", "market_prob_away", "market_overround",
    "market_elo_delta",
]

# --- Odds API ---
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds/"
ODDS_CACHE_TTL = 900  # 15 minutes in seconds
AU_BOOKMAKERS = [
    "sportsbet", "tab", "pointsbetau", "unibet", "ladbrokes_au",
    "betfair_ex_au", "neds", "bluebet",
]

# --- Database ---
DB_PATH = os.path.join(os.path.dirname(__file__), "bets.db")
