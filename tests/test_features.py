"""
Tests for feature engineering modules.

These tests verify that your feature calculations are correct.
Run them every time you change feature code to catch bugs early.

HOW TO RUN:
    pytest tests/test_features.py -v

BEGINNER NOTES:
    - Each function starting with "test_" is a separate test
    - assert checks if something is True; if not, the test fails
    - We create fake data to test with (no need for real IPL data)
    - pytest discovers and runs all test functions automatically
"""

import numpy as np
import pandas as pd
import pytest


def create_sample_deliveries():
    """Create a small sample delivery DataFrame for testing."""
    data = {
        "match_id": ["m1"] * 12 + ["m2"] * 12,
        "innings": [1] * 6 + [2] * 6 + [1] * 6 + [2] * 6,
        "batting_team": ["CSK"] * 6 + ["MI"] * 6 + ["CSK"] * 6 + ["RCB"] * 6,
        "over": [0, 0, 1, 1, 16, 16] * 4,
        "ball": [1, 2, 1, 2, 1, 2] * 4,
        "batter": ["Kohli", "Kohli", "Rohit", "Rohit", "Dhoni", "Dhoni"] * 4,
        "bowler": ["Bumrah", "Bumrah", "Ashwin", "Ashwin", "Bumrah", "Bumrah"] * 4,
        "non_striker": ["Rohit", "Rohit", "Kohli", "Kohli", "Jadeja", "Jadeja"] * 4,
        "runs_batter": [4, 1, 6, 0, 2, 4, 1, 0, 4, 2, 6, 1, 4, 1, 6, 0, 2, 4, 1, 0, 4, 2, 6, 1],
        "runs_extras": [0] * 24,
        "runs_total": [4, 1, 6, 0, 2, 4, 1, 0, 4, 2, 6, 1, 4, 1, 6, 0, 2, 4, 1, 0, 4, 2, 6, 1],
        "extras_wides": [0] * 24,
        "extras_noballs": [0] * 24,
        "extras_byes": [0] * 24,
        "extras_legbyes": [0] * 24,
        "extras_penalty": [0] * 24,
        "is_wicket": [0, 0, 0, 1, 0, 0] * 4,
        "wicket_kind": ["", "", "", "bowled", "", ""] * 4,
        "wicket_player": ["", "", "", "Rohit", "", ""] * 4,
        "wicket_fielders": [""] * 24,
        "phase": ["powerplay", "powerplay", "powerplay", "powerplay", "death", "death"] * 4,
    }
    return pd.DataFrame(data)


def create_sample_matches():
    """Create a small sample matches DataFrame for testing."""
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3", "m4", "m5"],
        "season": ["2023", "2023", "2023", "2023", "2023"],
        "date": pd.to_datetime(["2023-04-01", "2023-04-03", "2023-04-05", "2023-04-07", "2023-04-09"]),
        "team1": ["CSK", "MI", "CSK", "RCB", "CSK"],
        "team2": ["MI", "RCB", "RCB", "MI", "MI"],
        "venue": ["Chepauk", "Wankhede", "Chinnaswamy", "Wankhede", "Chepauk"],
        "city": ["Chennai", "Mumbai", "Bengaluru", "Mumbai", "Chennai"],
        "toss_winner": ["CSK", "MI", "RCB", "MI", "CSK"],
        "toss_decision": ["field", "bat", "field", "field", "bat"],
        "winner": ["CSK", "MI", "CSK", "MI", "CSK"],
        "win_by_runs": [20, 0, 15, 0, 30],
        "win_by_wickets": [0, 6, 0, 5, 0],
        "player_of_match": ["Dhoni", "Rohit", "Kohli", "Bumrah", "Jadeja"],
    })


class TestPlayerForm:
    """Tests for player form calculations."""

    def test_batting_form_basic(self):
        """Test that batting form returns expected structure."""
        from src.features.player_form import calculate_batting_form

        deliveries = create_sample_deliveries()
        form = calculate_batting_form(deliveries, "Kohli", window=5)

        assert "form_index" in form
        assert "weighted_avg" in form
        assert "weighted_sr" in form
        assert form["innings_count"] >= 0
        assert form["form_index"] >= 0

    def test_batting_form_unknown_player(self):
        """Test that unknown player returns zero form."""
        from src.features.player_form import calculate_batting_form

        deliveries = create_sample_deliveries()
        form = calculate_batting_form(deliveries, "Unknown Player XYZ")

        assert form["form_index"] == 0
        assert form["innings_count"] == 0

    def test_bowling_form_basic(self):
        """Test that bowling form returns expected structure."""
        from src.features.player_form import calculate_bowling_form

        deliveries = create_sample_deliveries()
        form = calculate_bowling_form(deliveries, "Bumrah", window=5)

        assert "form_index" in form
        assert "weighted_economy" in form
        assert form["match_count"] >= 0


class TestHeadToHead:
    """Tests for head-to-head calculations."""

    def test_h2h_basic(self):
        """Test basic H2H calculation."""
        from src.features.head_to_head import calculate_team_h2h

        matches = create_sample_matches()
        h2h = calculate_team_h2h(matches, "CSK", "MI")

        assert h2h["h2h_total_matches"] >= 0
        assert 0 <= h2h["h2h_team1_win_pct"] <= 1
        assert h2h["h2h_team1_wins"] + h2h["h2h_team2_wins"] <= h2h["h2h_total_matches"]

    def test_h2h_no_matches(self):
        """Test H2H when teams have never played."""
        from src.features.head_to_head import calculate_team_h2h

        matches = create_sample_matches()
        h2h = calculate_team_h2h(matches, "CSK", "SRH")  # SRH not in sample data

        assert h2h["h2h_total_matches"] == 0
        assert h2h["h2h_team1_win_pct"] == 0.5  # Default


class TestTeamStrength:
    """Tests for team strength calculations."""

    def test_batting_depth_score_range(self):
        """Test that batting depth score is between 0 and 100."""
        from src.features.team_strength import calculate_team_batting_depth

        batting = pd.DataFrame({
            "batter": ["A", "B", "C", "D", "E", "F", "G"],
            "batting_average": [35, 30, 28, 25, 22, 20, 18],
            "strike_rate": [140, 135, 130, 125, 120, 115, 110],
        })
        score = calculate_team_batting_depth(batting)
        assert 0 <= score <= 100

    def test_empty_team_returns_zero(self):
        """Test that empty team data returns 0."""
        from src.features.team_strength import calculate_team_batting_depth

        empty = pd.DataFrame(columns=["batter", "batting_average", "strike_rate"])
        score = calculate_team_batting_depth(empty)
        assert score == 0


class TestVenueFeatures:
    """Tests for venue feature calculations."""

    def test_venue_stats_structure(self):
        """Test that venue stats returns expected columns."""
        from src.features.venue_features import calculate_venue_stats

        matches = create_sample_matches()
        deliveries = create_sample_deliveries()
        venue_stats = calculate_venue_stats(matches, deliveries)

        assert "venue" in venue_stats.columns
        assert len(venue_stats) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
