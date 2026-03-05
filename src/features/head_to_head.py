"""
Head-to-Head (H2H) analysis between teams and key player matchups.

Some teams historically dominate others (e.g., CSK vs MI rivalry).
Some batsmen struggle against specific bowlers. This module captures
those patterns as features for the prediction model.

HOW IT WORKS:
    - Team vs Team: Win %, average margin, recent form against each other
    - Batter vs Bowler: Dismissal rate, runs scored, strike rate
    - Venue-specific H2H: How teams perform against each other at specific grounds
"""

import numpy as np
import pandas as pd


def calculate_team_h2h(matches: pd.DataFrame, team1: str, team2: str) -> dict:
    """
    Calculate head-to-head record between two teams.

    Args:
        matches: Match-level DataFrame with columns [team1, team2, winner, ...]
        team1: First team name
        team2: Second team name

    Returns:
        Dictionary with H2H metrics
    """
    # Filter matches between these two teams
    h2h_matches = matches[
        ((matches["team1"] == team1) & (matches["team2"] == team2)) |
        ((matches["team1"] == team2) & (matches["team2"] == team1))
    ].copy()

    total = len(h2h_matches)
    if total == 0:
        return {
            "h2h_total_matches": 0,
            "h2h_team1_wins": 0,
            "h2h_team2_wins": 0,
            "h2h_team1_win_pct": 0.5,
            "h2h_team2_win_pct": 0.5,
            "h2h_no_results": 0,
            "h2h_recent5_team1_wins": 0,
            "h2h_avg_win_margin_runs": 0,
            "h2h_avg_win_margin_wickets": 0,
        }

    team1_wins = len(h2h_matches[h2h_matches["winner"] == team1])
    team2_wins = len(h2h_matches[h2h_matches["winner"] == team2])
    no_results = total - team1_wins - team2_wins

    # Recent 5 matches
    recent5 = h2h_matches.sort_values("date", ascending=False).head(5)
    recent5_team1_wins = len(recent5[recent5["winner"] == team1])

    # Average win margins
    team1_win_matches = h2h_matches[h2h_matches["winner"] == team1]
    avg_run_margin = team1_win_matches["win_by_runs"].mean() if len(team1_win_matches) > 0 else 0
    avg_wkt_margin = team1_win_matches["win_by_wickets"].mean() if len(team1_win_matches) > 0 else 0

    return {
        "h2h_total_matches": total,
        "h2h_team1_wins": team1_wins,
        "h2h_team2_wins": team2_wins,
        "h2h_team1_win_pct": round(team1_wins / max(total, 1), 4),
        "h2h_team2_win_pct": round(team2_wins / max(total, 1), 4),
        "h2h_no_results": no_results,
        "h2h_recent5_team1_wins": recent5_team1_wins,
        "h2h_avg_win_margin_runs": round(avg_run_margin, 2),
        "h2h_avg_win_margin_wickets": round(avg_wkt_margin, 2),
    }


def calculate_venue_h2h(
    matches: pd.DataFrame, team1: str, team2: str, venue: str
) -> dict:
    """Calculate H2H record at a specific venue."""
    venue_matches = matches[matches["venue"] == venue]
    h2h = venue_matches[
        ((venue_matches["team1"] == team1) & (venue_matches["team2"] == team2)) |
        ((venue_matches["team1"] == team2) & (venue_matches["team2"] == team1))
    ]

    total = len(h2h)
    team1_wins = len(h2h[h2h["winner"] == team1])

    return {
        "venue_h2h_matches": total,
        "venue_h2h_team1_wins": team1_wins,
        "venue_h2h_team1_win_pct": round(team1_wins / max(total, 1), 4),
    }


def calculate_batter_vs_bowler(
    deliveries: pd.DataFrame, batter: str, bowler: str
) -> dict:
    """
    Calculate how a specific batter performs against a specific bowler.

    This is crucial for player matchup analysis. E.g., "How does Kohli
    perform against Bumrah?"
    """
    matchup = deliveries[
        (deliveries["batter"] == batter) & (deliveries["bowler"] == bowler)
    ]

    balls = len(matchup)
    if balls == 0:
        return {
            "matchup_balls": 0,
            "matchup_runs": 0,
            "matchup_sr": 0,
            "matchup_dismissals": 0,
            "matchup_dot_pct": 0,
        }

    runs = matchup["runs_batter"].sum()
    dismissals = matchup["is_wicket"].sum()
    dots = (matchup["runs_total"] == 0).sum()

    return {
        "matchup_balls": balls,
        "matchup_runs": int(runs),
        "matchup_sr": round((runs / balls) * 100, 2),
        "matchup_dismissals": int(dismissals),
        "matchup_dot_pct": round((dots / balls) * 100, 2),
    }


def build_all_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build H2H features for every team pair in the dataset.

    Returns a DataFrame that can be merged with match features.
    """
    print("  Building head-to-head features...")

    teams = list(set(matches["team1"].unique()) | set(matches["team2"].unique()))
    records = []

    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            h2h = calculate_team_h2h(matches, t1, t2)
            h2h["team1"] = t1
            h2h["team2"] = t2
            records.append(h2h)

            # Also add the reverse
            h2h_rev = calculate_team_h2h(matches, t2, t1)
            h2h_rev["team1"] = t2
            h2h_rev["team2"] = t1
            records.append(h2h_rev)

    df = pd.DataFrame(records)
    print(f"    Built H2H for {len(df)} team pairs")
    return df
