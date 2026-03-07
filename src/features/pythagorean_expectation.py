"""
Phase E: Pythagorean Win Expectation for T20 Cricket

Estimates a team's "true" win rate based on runs scored vs runs conceded,
rather than actual wins/losses. Teams whose actual win% exceeds their
Pythagorean expectation are "lucky" and likely to regress; teams below
are "unlucky" and likely to improve.

Formula: Win% = RS^x / (RS^x + RC^x)  where x ≈ 6.5 for T20 cricket.

The exponent x = 6.5 is derived from T20 scoring patterns — the higher
exponent (vs baseball's ~2) reflects T20's narrower run margins and
high-scoring nature.

HOW TO USE:
    from pythagorean_expectation import calculate_pythagorean_expectation
    result = calculate_pythagorean_expectation(team, matches, deliveries, before_date)
"""

import pandas as pd
import numpy as np


def calculate_pythagorean_expectation(
    team: str,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    before_date,
    exponent: float = 6.5,
    lookback_matches: int = 0,
) -> dict:
    """
    Calculate Pythagorean Win Expectation for a team using deliveries data.

    Args:
        team: Team name (standardised)
        matches: DataFrame with columns [match_id, date, team1, team2, winner]
        deliveries: DataFrame with columns [match_id, batting_team, runs_total]
        before_date: Only use matches before this date
        exponent: Pythagorean exponent (6.5 for T20)
        lookback_matches: If > 0, only use last N matches; 0 = use current season

    Returns:
        dict with keys:
            pwe_expected_win_pct: float (Pythagorean expected win %)
            pwe_actual_win_pct: float (actual win %)
            pwe_performance_diff: float (actual - expected; positive = overperforming)
            runs_scored_per_match: float
            runs_conceded_per_match: float
            sample_size: int
    """
    default = {
        "pwe_expected_win_pct": 0.5,
        "pwe_actual_win_pct": 0.5,
        "pwe_performance_diff": 0.0,
        "runs_scored_per_match": 0.0,
        "runs_conceded_per_match": 0.0,
        "sample_size": 0,
    }

    # Get team's matches before the cutoff date
    team_matches = matches[
        ((matches["team1"] == team) | (matches["team2"] == team))
        & (matches["date"] < before_date)
    ].sort_values("date", ascending=False)

    if lookback_matches > 0:
        team_matches = team_matches.head(lookback_matches)

    if len(team_matches) < 3:
        return default

    match_ids = set(team_matches["match_id"].tolist())

    # Calculate runs scored and runs conceded from deliveries
    match_deliveries = deliveries[deliveries["match_id"].isin(match_ids)]
    if match_deliveries.empty:
        return default

    # Runs scored = sum of runs_total when team is batting
    batting_deliveries = match_deliveries[match_deliveries["batting_team"] == team]
    runs_scored = batting_deliveries["runs_total"].sum() if not batting_deliveries.empty else 0

    # Runs conceded = sum of runs_total when team is NOT batting (but in their matches)
    bowling_deliveries = match_deliveries[match_deliveries["batting_team"] != team]
    runs_conceded = bowling_deliveries["runs_total"].sum() if not bowling_deliveries.empty else 0

    if runs_scored == 0 and runs_conceded == 0:
        return default

    # Count innings for per-match averages
    num_matches = len(team_matches)

    # Pythagorean formula
    rs_exp = runs_scored ** exponent
    rc_exp = runs_conceded ** exponent

    if rs_exp + rc_exp == 0:
        pwe = 0.5
    else:
        pwe = rs_exp / (rs_exp + rc_exp)

    # Actual win %
    wins = len(team_matches[team_matches["winner"] == team])
    valid_matches = len(team_matches[~team_matches["winner"].isin(["no result", "tie", ""])])
    actual_win_pct = wins / max(valid_matches, 1)

    return {
        "pwe_expected_win_pct": round(pwe, 4),
        "pwe_actual_win_pct": round(actual_win_pct, 4),
        "pwe_performance_diff": round(actual_win_pct - pwe, 4),
        "runs_scored_per_match": round(runs_scored / num_matches, 1),
        "runs_conceded_per_match": round(runs_conceded / num_matches, 1),
        "sample_size": num_matches,
    }


def calculate_all_teams_pwe(
    teams: list,
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    before_date=None,
    exponent: float = 6.5,
) -> list:
    """
    Calculate Pythagorean Win Expectation for all teams.
    Returns a list of dicts sorted by performance_diff (most overperforming first).

    Used by the dashboard to show the Expected vs Actual chart.
    """
    if before_date is None:
        before_date = pd.Timestamp("2030-01-01")

    results = []
    for team in teams:
        pwe = calculate_pythagorean_expectation(
            team, matches, deliveries, before_date, exponent
        )
        pwe["team"] = team
        results.append(pwe)

    # Sort by performance diff (biggest overperformers first)
    results.sort(key=lambda x: x["pwe_performance_diff"], reverse=True)
    return results


if __name__ == "__main__":
    print("Pythagorean Win Expectation module loaded.")
    print("  Formula: Win% = RS^6.5 / (RS^6.5 + RC^6.5)")
    print("  Use calculate_pythagorean_expectation() for a single team.")
    print("  Use calculate_all_teams_pwe() for all teams.")
