"""
Phase D: Live Score Collector — CricketData.org (formerly CricAPI) Integration
Fetches real-time match scores during live IPL matches.

Free tier: 100 requests/day. We track ONE match at a time,
polling every 30 seconds during a live match (~120 requests per 3-hour match).

HOW TO USE:
    from src.data_collection.live_score_collector import LiveScoreCollector
    collector = LiveScoreCollector()
    matches = collector.get_live_matches()
    state = collector.get_match_state(match_id)

SETUP:
    1. Sign up at https://cricketdata.org (free)
    2. Add CRICAPI_KEY=your_key to .env file
"""

import os
import time
import json
from datetime import datetime, date
from typing import Optional, Dict, List, Any

import requests
from dotenv import load_dotenv

load_dotenv()

CRICAPI_KEY = os.getenv("CRICAPI_KEY", "")
BASE_URL = "https://api.cricapi.com/v1"

# Rate limit tracking
_daily_requests = {"count": 0, "date": None}
_DAILY_LIMIT = 90  # Leave buffer below 100

# Response cache
_cache = {"data": None, "timestamp": 0, "ttl": 15}  # 15-second cache


class MatchState:
    """Structured representation of a live match state."""

    def __init__(
        self,
        match_id: str,
        name: str,
        status: str,
        venue: str,
        team1: str,
        team2: str,
        score: List[Dict],
        team1_score: str = "",
        team2_score: str = "",
        current_batting: str = "",
        match_started: bool = False,
        match_ended: bool = False,
        date_str: str = "",
    ):
        self.match_id = match_id
        self.name = name
        self.status = status
        self.venue = venue
        self.team1 = team1
        self.team2 = team2
        self.score = score  # Raw score data from API
        self.team1_score = team1_score
        self.team2_score = team2_score
        self.current_batting = current_batting
        self.match_started = match_started
        self.match_ended = match_ended
        self.date_str = date_str
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        # Parse score strings for structured data
        t1_runs, t1_wickets, t1_overs = self._parse_score(self.team1_score)
        t2_runs, t2_wickets, t2_overs = self._parse_score(self.team2_score)

        # Calculate run rates
        t1_rr = round(t1_runs / t1_overs, 2) if t1_overs > 0 else 0
        t2_rr = round(t2_runs / t2_overs, 2) if t2_overs > 0 else 0

        # Determine innings and target
        innings = 1
        target = 0
        required_run_rate = 0
        if t2_overs > 0 or (self.match_started and t1_wickets == 10):
            innings = 2
            target = t1_runs + 1
            remaining_runs = target - t2_runs
            remaining_overs = 20.0 - t2_overs
            if remaining_overs > 0:
                required_run_rate = round(remaining_runs / remaining_overs, 2)

        return {
            "match_id": self.match_id,
            "name": self.name,
            "status": self.status,
            "venue": self.venue,
            "team1": self.team1,
            "team2": self.team2,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "team1_runs": t1_runs,
            "team1_wickets": t1_wickets,
            "team1_overs": t1_overs,
            "team2_runs": t2_runs,
            "team2_wickets": t2_wickets,
            "team2_overs": t2_overs,
            "team1_run_rate": t1_rr,
            "team2_run_rate": t2_rr,
            "innings": innings,
            "target": target,
            "required_run_rate": required_run_rate,
            "current_batting": self.current_batting,
            "match_started": self.match_started,
            "match_ended": self.match_ended,
            "date": self.date_str,
            "last_updated": self.last_updated,
        }

    @staticmethod
    def _parse_score(score_str: str):
        """Parse '185/4 (16.3)' into (runs, wickets, overs)."""
        if not score_str:
            return 0, 0, 0.0
        try:
            # Handle formats: "185/4 (16.3)", "185/4", "185"
            parts = score_str.strip().split("(")
            overs = 0.0
            if len(parts) > 1:
                overs = float(parts[1].replace(")", "").strip())

            score_part = parts[0].strip()
            if "/" in score_part:
                runs, wickets = score_part.split("/")
                return int(runs.strip()), int(wickets.strip()), overs
            else:
                return int(score_part.strip()), 0, overs
        except (ValueError, IndexError):
            return 0, 0, 0.0


class LiveScoreCollector:
    """Fetches live cricket match data from CricAPI."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or CRICAPI_KEY
        if not self.api_key:
            print("WARNING: CRICAPI_KEY not set in .env file")

    def _check_rate_limit(self) -> bool:
        """Check if we've exceeded daily request limit."""
        global _daily_requests
        today = date.today().isoformat()

        if _daily_requests["date"] != today:
            _daily_requests = {"count": 0, "date": today}

        if _daily_requests["count"] >= _DAILY_LIMIT:
            print(f"Rate limit reached: {_daily_requests['count']}/{_DAILY_LIMIT} requests today")
            return False
        return True

    def _make_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make authenticated API request with rate limiting."""
        if not self.api_key:
            return None

        if not self._check_rate_limit():
            return None

        url = f"{BASE_URL}/{endpoint}"
        all_params = {"apikey": self.api_key}
        if params:
            all_params.update(params)

        try:
            response = requests.get(url, params=all_params, timeout=10)
            _daily_requests["count"] += 1

            if response.status_code == 429:
                print("CricAPI rate limit hit (429). Backing off.")
                return None

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return data
                else:
                    print(f"CricAPI error: {data.get('status')}")
                    return None

            print(f"CricAPI HTTP {response.status_code}")
            return None

        except requests.Timeout:
            print("CricAPI request timed out")
            return None
        except requests.RequestException as e:
            print(f"CricAPI request failed: {e}")
            return None

    def get_live_matches(self) -> List[dict]:
        """
        Fetch all current/upcoming IPL matches.
        Returns list of match dicts with basic info.
        """
        global _cache

        # Check cache
        now = time.time()
        if _cache["data"] is not None and (now - _cache["timestamp"]) < _cache["ttl"]:
            return _cache["data"]

        data = self._make_request("currentMatches", {"offset": 0})
        if not data or "data" not in data:
            # Return cached data if available
            return _cache["data"] or []

        # Filter for IPL matches only
        ipl_matches = []
        for match in data.get("data", []):
            match_type = (match.get("matchType", "") or "").lower()
            name = (match.get("name", "") or "").lower()
            series = (match.get("series", "") or "").lower()

            # Check if it's an IPL match
            if "ipl" in series or "indian premier league" in series or "ipl" in name:
                state = self._parse_match(match)
                if state:
                    ipl_matches.append(state.to_dict())

        _cache = {"data": ipl_matches, "timestamp": now, "ttl": 15}
        return ipl_matches

    def get_match_state(self, match_id: str) -> Optional[dict]:
        """Fetch detailed live state for a specific match."""
        data = self._make_request("match_info", {"id": match_id})
        if not data or "data" not in data:
            return None

        state = self._parse_match(data["data"])
        return state.to_dict() if state else None

    def _parse_match(self, match_data: dict) -> Optional[MatchState]:
        """Parse raw CricAPI response into MatchState."""
        try:
            match_id = match_data.get("id", "")
            name = match_data.get("name", "")
            status = match_data.get("status", "")
            venue = match_data.get("venue", "")
            date_str = match_data.get("date", "")

            teams = match_data.get("teams", [])
            team1 = teams[0] if len(teams) > 0 else ""
            team2 = teams[1] if len(teams) > 1 else ""

            # Parse scores
            score_data = match_data.get("score", [])
            team1_score = ""
            team2_score = ""

            for s in score_data:
                inning = s.get("inning", "")
                runs = s.get("r", 0)
                wickets = s.get("w", 0)
                overs = s.get("o", 0)
                score_str = f"{runs}/{wickets} ({overs})"

                if team1.lower() in inning.lower():
                    team1_score = score_str
                elif team2.lower() in inning.lower():
                    team2_score = score_str

            match_started = match_data.get("matchStarted", False)
            match_ended = match_data.get("matchEnded", False)

            return MatchState(
                match_id=match_id,
                name=name,
                status=status,
                venue=venue,
                team1=team1,
                team2=team2,
                score=score_data,
                team1_score=team1_score,
                team2_score=team2_score,
                current_batting=team1 if not team2_score else team2,
                match_started=match_started,
                match_ended=match_ended,
                date_str=date_str,
            )
        except Exception as e:
            print(f"Error parsing match: {e}")
            return None

    def get_requests_remaining(self) -> int:
        """Check how many API requests remain today."""
        today = date.today().isoformat()
        if _daily_requests["date"] != today:
            return _DAILY_LIMIT
        return max(0, _DAILY_LIMIT - _daily_requests["count"])


# ── Demo / Test ──
if __name__ == "__main__":
    collector = LiveScoreCollector()
    print(f"Requests remaining today: {collector.get_requests_remaining()}")

    matches = collector.get_live_matches()
    if matches:
        print(f"\nFound {len(matches)} IPL match(es):")
        for m in matches:
            print(f"  {m['name']} — {m['status']}")
            if m["team1_score"]:
                print(f"    {m['team1']}: {m['team1_score']}")
            if m["team2_score"]:
                print(f"    {m['team2']}: {m['team2_score']}")
    else:
        print("\nNo live IPL matches right now.")
        print("(This is normal when no match is being played)")
