"""
Tests for team name standardisation.

Ensures all historical team names are correctly mapped to current names.

HOW TO RUN:
    pytest tests/test_team_name_mapper.py -v
"""

import os
import sys

import pytest

# Add project root and data_collection to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "data_collection")))

from team_name_mapper import (
    standardise_team_name,
    get_short_code,
    apply_team_mapping,
    ACTIVE_TEAMS,
    TEAM_NAME_MAP,
    TEAM_SHORT_CODES,
)


class TestStandardiseTeamName:
    """Tests for the standardise_team_name function."""

    def test_delhi_daredevils_to_capitals(self):
        assert standardise_team_name("Delhi Daredevils") == "Delhi Capitals"

    def test_kings_xi_punjab_to_punjab_kings(self):
        assert standardise_team_name("Kings XI Punjab") == "Punjab Kings"

    def test_rcb_bangalore_to_bengaluru(self):
        assert standardise_team_name("Royal Challengers Bangalore") == "Royal Challengers Bengaluru"

    def test_rising_pune_supergiants_plural(self):
        """Both singular and plural should map to the same name."""
        assert standardise_team_name("Rising Pune Supergiants") == "Rising Pune Supergiant"
        assert standardise_team_name("Rising Pune Supergiant") == "Rising Pune Supergiant"

    def test_current_names_unchanged(self):
        """Current team names should stay the same."""
        for team in ACTIVE_TEAMS:
            assert standardise_team_name(team) == team

    def test_case_insensitive(self):
        assert standardise_team_name("delhi daredevils") == "Delhi Capitals"
        assert standardise_team_name("KINGS XI PUNJAB") == "Punjab Kings"

    def test_empty_and_none(self):
        assert standardise_team_name("") == ""
        assert standardise_team_name(None) is None

    def test_unknown_team_returned_as_is(self):
        assert standardise_team_name("Unknown Team XYZ") == "Unknown Team XYZ"


class TestShortCodes:
    """Tests for short code lookups."""

    def test_csk(self):
        assert get_short_code("Chennai Super Kings") == "CSK"

    def test_old_name_gives_current_code(self):
        assert get_short_code("Delhi Daredevils") == "DC"
        assert get_short_code("Kings XI Punjab") == "PBKS"
        assert get_short_code("Royal Challengers Bangalore") == "RCB"

    def test_all_active_teams_have_codes(self):
        for team in ACTIVE_TEAMS:
            code = get_short_code(team)
            assert len(code) >= 2, f"{team} has no short code"


class TestApplyTeamMapping:
    """Tests for DataFrame-level mapping."""

    def test_basic_mapping(self):
        import pandas as pd

        df = pd.DataFrame({
            "team1": ["Delhi Daredevils", "Kings XI Punjab"],
            "team2": ["Chennai Super Kings", "Royal Challengers Bangalore"],
            "winner": ["Delhi Daredevils", "Royal Challengers Bangalore"],
        })

        result = apply_team_mapping(df, columns=["team1", "team2", "winner"])

        assert result["team1"].iloc[0] == "Delhi Capitals"
        assert result["team1"].iloc[1] == "Punjab Kings"
        assert result["team2"].iloc[1] == "Royal Challengers Bengaluru"
        assert result["winner"].iloc[0] == "Delhi Capitals"

    def test_auto_detect_columns(self):
        import pandas as pd

        df = pd.DataFrame({
            "team1": ["Delhi Daredevils"],
            "team2": ["Mumbai Indians"],
            "winner": ["Delhi Daredevils"],
            "toss_winner": ["Mumbai Indians"],
            "batting_team": ["Delhi Daredevils"],
            "other_column": ["keep this"],
        })

        result = apply_team_mapping(df)  # No columns specified — auto-detect

        assert result["team1"].iloc[0] == "Delhi Capitals"
        assert result["batting_team"].iloc[0] == "Delhi Capitals"
        assert result["other_column"].iloc[0] == "keep this"  # Not touched


class TestConsistency:
    """Ensure internal consistency of the mapping data."""

    def test_all_active_teams_in_map(self):
        for team in ACTIVE_TEAMS:
            assert team in TEAM_NAME_MAP, f"{team} missing from TEAM_NAME_MAP"

    def test_all_active_teams_have_short_codes(self):
        for team in ACTIVE_TEAMS:
            assert team in TEAM_SHORT_CODES, f"{team} missing from TEAM_SHORT_CODES"

    def test_exactly_10_active_teams(self):
        assert len(ACTIVE_TEAMS) == 10

    def test_map_values_are_in_short_codes(self):
        """Every mapped-to team name should have a short code."""
        unique_targets = set(TEAM_NAME_MAP.values())
        for target in unique_targets:
            assert target in TEAM_SHORT_CODES, f"{target} has no short code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
