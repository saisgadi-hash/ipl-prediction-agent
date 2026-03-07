"""
Phase D: Betting Odds Collector — The-Odds-API Integration
Fetches pre-match and live betting odds for IPL matches.

Free tier: 500 requests/month. We poll daily for pre-match odds
and make sparse live checks during matches.

HOW TO USE:
    from src.data_collection.odds_collector import OddsCollector
    collector = OddsCollector()
    odds = collector.get_upcoming_odds()

SETUP:
    1. Sign up at https://the-odds-api.com (free)
    2. Add THE_ODDS_API_KEY=your_key to .env file
"""

import os
import time
from datetime import datetime, date
from typing import Optional, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "cricket_ipl"  # The-Odds-API sport key for IPL

# Rate limit tracking
_monthly_requests = {"count": 0, "month": None}
_MONTHLY_LIMIT = 450  # Leave buffer below 500

# Cache
_odds_cache = {"data": None, "timestamp": 0, "ttl": 3600}  # 1-hour cache


class MatchOdds:
    """Structured representation of betting odds for a match."""

    def __init__(
        self,
        match_id: str,
        team1: str,
        team2: str,
        commence_time: str,
        bookmakers: List[Dict],
    ):
        self.match_id = match_id
        self.team1 = team1
        self.team2 = team2
        self.commence_time = commence_time
        self.bookmakers = bookmakers

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        # Calculate average implied probabilities across bookmakers
        t1_probs = []
        t2_probs = []

        bookmaker_data = {}
        for bm in self.bookmakers:
            bm_name = bm.get("title", "Unknown")
            outcomes = {}
            for market in bm.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 2.0)
                        implied_prob = round(1 / price, 4) if price > 0 else 0
                        outcomes[name] = {
                            "decimal_odds": price,
                            "implied_probability": implied_prob,
                        }
                        if name == self.team1:
                            t1_probs.append(implied_prob)
                        elif name == self.team2:
                            t2_probs.append(implied_prob)

            if outcomes:
                bookmaker_data[bm_name] = outcomes

        # Average implied probabilities (with overround removed)
        avg_t1 = sum(t1_probs) / len(t1_probs) if t1_probs else 0.5
        avg_t2 = sum(t2_probs) / len(t2_probs) if t2_probs else 0.5

        # Normalise to remove overround (bookmaker margin)
        total = avg_t1 + avg_t2
        if total > 0:
            avg_t1 = round(avg_t1 / total, 4)
            avg_t2 = round(avg_t2 / total, 4)

        return {
            "match_id": self.match_id,
            "team1": self.team1,
            "team2": self.team2,
            "commence_time": self.commence_time,
            "implied_probability_team1": avg_t1,
            "implied_probability_team2": avg_t2,
            "bookmakers": bookmaker_data,
            "num_bookmakers": len(bookmaker_data),
            "last_updated": datetime.now().isoformat(),
        }


class OddsCollector:
    """Fetches betting odds from The-Odds-API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or THE_ODDS_API_KEY
        if not self.api_key:
            print("WARNING: THE_ODDS_API_KEY not set in .env file")

    def _check_rate_limit(self) -> bool:
        """Check if we've exceeded monthly request limit."""
        global _monthly_requests
        current_month = date.today().strftime("%Y-%m")

        if _monthly_requests["month"] != current_month:
            _monthly_requests = {"count": 0, "month": current_month}

        if _monthly_requests["count"] >= _MONTHLY_LIMIT:
            print(f"Monthly rate limit reached: {_monthly_requests['count']}/{_MONTHLY_LIMIT}")
            return False
        return True

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make authenticated API request."""
        if not self.api_key:
            return None

        if not self._check_rate_limit():
            return None

        url = f"{BASE_URL}/{endpoint}"
        all_params = {"apiKey": self.api_key}
        if params:
            all_params.update(params)

        try:
            response = requests.get(url, params=all_params, timeout=10)
            _monthly_requests["count"] += 1

            # Track remaining requests from response headers
            remaining = response.headers.get("x-requests-remaining")
            if remaining:
                print(f"  Odds API requests remaining this month: {remaining}")

            if response.status_code == 401:
                print("Invalid Odds API key")
                return None
            if response.status_code == 429:
                print("Odds API rate limit hit")
                return None
            if response.status_code == 200:
                return response.json()

            print(f"Odds API HTTP {response.status_code}")
            return None

        except requests.Timeout:
            print("Odds API request timed out")
            return None
        except requests.RequestException as e:
            print(f"Odds API request failed: {e}")
            return None

    def get_upcoming_odds(self) -> List[dict]:
        """
        Fetch pre-match odds for upcoming IPL matches.
        Returns list of MatchOdds dicts.
        """
        global _odds_cache

        # Check cache
        now = time.time()
        if _odds_cache["data"] is not None and (now - _odds_cache["timestamp"]) < _odds_cache["ttl"]:
            return _odds_cache["data"]

        data = self._make_request(
            f"sports/{SPORT}/odds",
            {
                "regions": "eu,uk",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
        )

        if not data or not isinstance(data, list):
            return _odds_cache["data"] or []

        odds_list = []
        for event in data:
            match_odds = MatchOdds(
                match_id=event.get("id", ""),
                team1=event.get("home_team", ""),
                team2=event.get("away_team", ""),
                commence_time=event.get("commence_time", ""),
                bookmakers=event.get("bookmakers", []),
            )
            odds_list.append(match_odds.to_dict())

        _odds_cache = {"data": odds_list, "timestamp": now, "ttl": 3600}
        return odds_list

    def get_requests_remaining(self) -> int:
        """Check how many API requests remain this month."""
        current_month = date.today().strftime("%Y-%m")
        if _monthly_requests["month"] != current_month:
            return _MONTHLY_LIMIT
        return max(0, _MONTHLY_LIMIT - _monthly_requests["count"])


# ── Demo / Test ──
if __name__ == "__main__":
    collector = OddsCollector()
    print(f"Requests remaining this month: {collector.get_requests_remaining()}")

    odds = collector.get_upcoming_odds()
    if odds:
        print(f"\nFound odds for {len(odds)} match(es):")
        for o in odds:
            print(f"  {o['team1']} vs {o['team2']}")
            print(f"    Implied: {o['team1']} {o['implied_probability_team1']:.1%} | {o['team2']} {o['implied_probability_team2']:.1%}")
            print(f"    Bookmakers: {o['num_bookmakers']}")
    else:
        print("\nNo upcoming IPL odds available.")
        print("(Odds typically appear 1-2 days before a match)")
