"""Configuration loader for IPL Prediction Agent."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load settings.yaml
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
with open(SETTINGS_PATH, "r") as f:
    SETTINGS = yaml.safe_load(f)

# ── Data Directories ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── API Keys (from .env) ──
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/ipl_predictions.db")

# ── Model Config Shortcuts ──
MODEL_CONFIG = SETTINGS.get("model", {})
VENUES = SETTINGS.get("venues", [])
TEAMS = SETTINGS.get("teams", [])
DATA_SOURCES = SETTINGS.get("data_sources", {})


def get_venue_info(city_name: str) -> dict:
    """Look up venue details by city name."""
    for venue in VENUES:
        if venue["city"].lower() == city_name.lower():
            return venue
    return {}


def get_team_info(team_code: str) -> dict:
    """Look up team details by team code (e.g., 'CSK')."""
    for team in TEAMS:
        if team["code"].lower() == team_code.lower():
            return team
    return {}
