"""
Calculate player form indices with recency weighting.

Recent performance matters more than career averages. This module
computes form indices that exponentially weight recent innings,
so a player who scored 3 consecutive fifties ranks higher than
one who scored a century 2 months ago.

HOW IT WORKS:
    - Takes a player's last N innings (default: 10)
    - Applies exponential decay: most recent innings has weight 1.0,
      second-most-recent has weight 0.85, third has 0.72, etc.
    - Calculates weighted batting average, strike rate, and a composite form index
    - Same logic applies for bowling form (economy, wickets, dot ball %)

BEGINNER NOTES:
    - Exponential decay: each older innings is worth 85% of the previous one
    - np.average(values, weights=weights) calculates a weighted average
    - Form index combines multiple metrics into one number for easier comparison
"""

import numpy as np
import pandas as pd


def calculate_batting_form(
    deliveries: pd.DataFrame,
    player_name: str,
    window: int = 10,
    decay: float = 0.85,
) -> dict:
    """
    Calculate recency-weighted batting form for a player.

    Args:
        deliveries: Ball-by-ball DataFrame
        player_name: Name of the batter
        window: Number of recent innings to consider
        decay: Exponential decay factor (0.85 = each older innings worth 85% less)

    Returns:
        Dictionary with form metrics:
        - form_index: Composite form score (0-100 scale)
        - weighted_avg: Recency-weighted batting average
        - weighted_sr: Recency-weighted strike rate
        - consistency: How consistent the recent innings are (lower = more consistent)
        - boundary_rate: Recent boundary percentage
        - innings_count: Number of innings in the window
    """
    # Get this player's innings
    player_balls = deliveries[deliveries["batter"] == player_name]

    if len(player_balls) == 0:
        return _empty_batting_form()

    # Aggregate per innings
    innings = player_balls.groupby(["match_id", "innings"]).agg(
        runs=("runs_batter", "sum"),
        balls_faced=("runs_batter", "count"),
        fours=("runs_batter", lambda x: (x == 4).sum()),
        sixes=("runs_batter", lambda x: (x == 6).sum()),
        dot_balls=("runs_batter", lambda x: (x == 0).sum()),
    ).reset_index()

    # Sort by match_id (proxy for date) and take most recent
    innings = innings.sort_values("match_id", ascending=False).head(window)

    if len(innings) == 0:
        return _empty_batting_form()

    # Create exponential weights
    n = len(innings)
    weights = np.array([decay ** i for i in range(n)])
    weights = weights / weights.sum()  # Normalise to sum to 1

    # Calculate weighted metrics
    weighted_avg = np.average(innings["runs"], weights=weights)

    # Strike rate per innings
    sr_per_innings = np.where(
        innings["balls_faced"] > 0,
        (innings["runs"] / innings["balls_faced"]) * 100,
        0
    )
    weighted_sr = np.average(sr_per_innings, weights=weights)

    # Boundary rate per innings
    boundary_rate_per = np.where(
        innings["balls_faced"] > 0,
        ((innings["fours"] + innings["sixes"]) / innings["balls_faced"]) * 100,
        0
    )
    weighted_boundary_rate = np.average(boundary_rate_per, weights=weights)

    # Consistency (coefficient of variation - lower is more consistent)
    consistency = innings["runs"].std() / max(innings["runs"].mean(), 1)

    # Composite form index (0-100 scale)
    # Weighted: 40% average, 30% strike rate, 20% boundary rate, 10% consistency
    form_index = (
        min(weighted_avg / 50, 1.0) * 40 +        # Normalised to 50-run benchmark
        min(weighted_sr / 160, 1.0) * 30 +         # Normalised to 160 SR benchmark
        min(weighted_boundary_rate / 25, 1.0) * 20 +  # Normalised to 25% boundary rate
        max(1 - consistency, 0) * 10                # Lower variance = higher score
    )

    return {
        "form_index": round(form_index, 2),
        "weighted_avg": round(weighted_avg, 2),
        "weighted_sr": round(weighted_sr, 2),
        "boundary_rate": round(weighted_boundary_rate, 2),
        "consistency": round(consistency, 2),
        "innings_count": n,
        "last_5_runs": innings["runs"].head(5).tolist(),
    }


def calculate_bowling_form(
    deliveries: pd.DataFrame,
    player_name: str,
    window: int = 10,
    decay: float = 0.85,
) -> dict:
    """
    Calculate recency-weighted bowling form for a player.

    Args:
        deliveries: Ball-by-ball DataFrame
        player_name: Name of the bowler
        window: Number of recent match appearances
        decay: Exponential decay factor

    Returns:
        Dictionary with bowling form metrics
    """
    player_balls = deliveries[deliveries["bowler"] == player_name]

    if len(player_balls) == 0:
        return _empty_bowling_form()

    # Aggregate per match
    match_bowling = player_balls.groupby("match_id").agg(
        runs_conceded=("runs_total", "sum"),
        balls_bowled=("bowler", "count"),
        wickets=("is_wicket", "sum"),
        dot_balls=("runs_total", lambda x: (x == 0).sum()),
        fours_conceded=("runs_batter", lambda x: (x == 4).sum()),
        sixes_conceded=("runs_batter", lambda x: (x == 6).sum()),
    ).reset_index()

    match_bowling = match_bowling.sort_values("match_id", ascending=False).head(window)

    if len(match_bowling) == 0:
        return _empty_bowling_form()

    n = len(match_bowling)
    weights = np.array([decay ** i for i in range(n)])
    weights = weights / weights.sum()

    # Economy per match
    economy_per_match = np.where(
        match_bowling["balls_bowled"] > 0,
        match_bowling["runs_conceded"] / (match_bowling["balls_bowled"] / 6),
        0
    )
    weighted_economy = np.average(economy_per_match, weights=weights)

    # Dot ball percentage per match
    dot_pct_per_match = np.where(
        match_bowling["balls_bowled"] > 0,
        (match_bowling["dot_balls"] / match_bowling["balls_bowled"]) * 100,
        0
    )
    weighted_dot_pct = np.average(dot_pct_per_match, weights=weights)

    weighted_wickets = np.average(match_bowling["wickets"], weights=weights)

    # Bowling form index (0-100, lower economy and more wickets = higher score)
    form_index = (
        max(1 - weighted_economy / 12, 0) * 35 +   # Economy: 12+ is terrible
        min(weighted_wickets / 3, 1.0) * 30 +        # Wickets: 3 per match is great
        min(weighted_dot_pct / 50, 1.0) * 25 +       # Dot ball %: 50% is excellent
        max(1 - match_bowling["runs_conceded"].std() / 20, 0) * 10  # Consistency
    )

    return {
        "form_index": round(form_index, 2),
        "weighted_economy": round(weighted_economy, 2),
        "weighted_wickets_per_match": round(weighted_wickets, 2),
        "weighted_dot_ball_pct": round(weighted_dot_pct, 2),
        "match_count": n,
        "last_5_wickets": match_bowling["wickets"].head(5).tolist(),
    }


def calculate_all_player_forms(
    deliveries: pd.DataFrame,
    window: int = 10,
    decay: float = 0.85,
) -> pd.DataFrame:
    """
    Calculate form indices for ALL players.

    Returns a DataFrame with one row per player containing both
    batting and bowling form metrics.
    """
    print("  Calculating player form indices...")

    # Get unique players (both batters and bowlers)
    batters = deliveries["batter"].unique()
    bowlers = deliveries["bowler"].unique()
    all_players = list(set(list(batters) + list(bowlers)))

    records = []
    for player in all_players:
        bat_form = calculate_batting_form(deliveries, player, window, decay)
        bowl_form = calculate_bowling_form(deliveries, player, window, decay)

        record = {"player": player}
        # Add batting form with prefix
        for k, v in bat_form.items():
            if k != "last_5_runs":
                record[f"bat_{k}"] = v
        # Add bowling form with prefix
        for k, v in bowl_form.items():
            if k != "last_5_wickets":
                record[f"bowl_{k}"] = v

        records.append(record)

    df = pd.DataFrame(records)
    print(f"    Calculated form for {len(df)} players")
    return df


def _empty_batting_form():
    return {
        "form_index": 0, "weighted_avg": 0, "weighted_sr": 0,
        "boundary_rate": 0, "consistency": 0, "innings_count": 0,
        "last_5_runs": [],
    }


def _empty_bowling_form():
    return {
        "form_index": 0, "weighted_economy": 0,
        "weighted_wickets_per_match": 0, "weighted_dot_ball_pct": 0,
        "match_count": 0, "last_5_wickets": [],
    }
