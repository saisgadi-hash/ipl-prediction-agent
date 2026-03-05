"""
Calculate Team Strength Index — a composite score for each team.

The Team Strength Index combines batting depth, bowling variety,
death-over capability, fielding rating, and bench strength into a
single number (0-100) that represents how strong a team is.

HOW IT WORKS:
    - Loads player stats for each team's squad
    - Scores each dimension (batting, bowling, fielding, etc.) on 0-100
    - Combines with configurable weights
    - Higher index = stronger team

BEGINNER NOTES:
    - We're building a "composite index" — one number from many inputs
    - Each component is normalised to 0-100 so they're comparable
    - Weights determine which components matter more (you can tune these)
"""

import numpy as np
import pandas as pd


def calculate_team_batting_depth(team_batting: pd.DataFrame) -> float:
    """
    Score the batting depth of a team (0-100).

    A team with 7 reliable batters scores higher than one
    that depends on 2-3 star players.

    BEGINNER NOTE:
        - "Batting depth" means how many batters can contribute runs
        - A batting average > 25 in T20s is considered solid
        - We look at the top 7 (typical batting positions 1-7)
    """
    if len(team_batting) == 0:
        return 0

    # Sort by batting average (best batters first)
    top7 = team_batting.nlargest(7, "batting_average")

    if len(top7) == 0:
        return 0

    # Average of top 7 batters' averages
    avg_of_top7 = top7["batting_average"].mean()

    # Strike rate bonus
    sr_bonus = max(0, (top7["strike_rate"].mean() - 120) / 50) * 20

    # Depth factor: penalise if only 3-4 good batters
    reliable_batters = len(top7[top7["batting_average"] > 20])
    depth_factor = min(reliable_batters / 7, 1.0)

    score = min(
        (avg_of_top7 / 35) * 60 +  # Normalise avg (35 is excellent in T20)
        sr_bonus +
        depth_factor * 20,
        100
    )
    return round(score, 2)


def calculate_team_bowling_strength(team_bowling: pd.DataFrame) -> float:
    """
    Score the bowling attack strength (0-100).

    Considers economy rate, wicket-taking ability, and variety
    (mix of pace and spin).
    """
    if len(team_bowling) == 0:
        return 0

    # Top 5 bowlers by wickets
    top5 = team_bowling.nlargest(5, "wickets").head(5)

    if len(top5) == 0:
        return 0

    # Economy score (lower is better)
    avg_economy = top5["economy"].mean()
    economy_score = max(0, (12 - avg_economy) / 5) * 40  # 7 econ = 40 pts

    # Wicket-taking score
    avg_wickets = top5["wickets"].mean()
    wicket_score = min(avg_wickets / 20, 1.0) * 30  # 20 wickets in season = full score

    # Dot ball score
    avg_dot_pct = top5["dot_ball_pct"].mean()
    dot_score = min(avg_dot_pct / 45, 1.0) * 20  # 45% dot balls = full score

    # Variety bonus (different bowling types)
    bowler_count = len(top5[top5["wickets"] >= 5])
    variety_score = min(bowler_count / 5, 1.0) * 10

    score = min(economy_score + wicket_score + dot_score + variety_score, 100)
    return round(score, 2)


def calculate_death_bowling_rating(team_bowling: pd.DataFrame) -> float:
    """
    Score death-over bowling capability (0-100).

    Death overs (16-20) are the most critical phase. A team that
    concedes < 10 runs per over in the death is elite.
    """
    if "death_economy" not in team_bowling.columns:
        return 50  # Default if phase data not available

    death_bowlers = team_bowling[team_bowling["death_economy"] > 0]
    if len(death_bowlers) == 0:
        return 50

    top3 = death_bowlers.nsmallest(3, "death_economy")
    avg_death_economy = top3["death_economy"].mean()

    # Score: economy of 8 = 100, economy of 14 = 0
    score = max(0, (14 - avg_death_economy) / 6) * 100
    return round(min(score, 100), 2)


def calculate_powerplay_rating(team_batting: pd.DataFrame, team_bowling: pd.DataFrame) -> float:
    """Score powerplay (overs 1-6) performance for both batting and bowling."""
    scores = []

    # Batting in powerplay
    if "powerplay_sr" in team_batting.columns:
        pp_batters = team_batting[team_batting["powerplay_balls"] > 20]
        if len(pp_batters) > 0:
            avg_pp_sr = pp_batters.nlargest(3, "powerplay_runs")["powerplay_sr"].mean()
            bat_score = min(avg_pp_sr / 160, 1.0) * 50
            scores.append(bat_score)

    # Bowling in powerplay
    if "powerplay_economy" in team_bowling.columns:
        pp_bowlers = team_bowling[team_bowling["powerplay_economy"] > 0]
        if len(pp_bowlers) > 0:
            avg_pp_econ = pp_bowlers.nsmallest(3, "powerplay_economy")["powerplay_economy"].mean()
            bowl_score = max(0, (10 - avg_pp_econ) / 4) * 50
            scores.append(bowl_score)

    return round(sum(scores), 2) if scores else 50


def calculate_team_strength_index(
    team_batting: pd.DataFrame,
    team_bowling: pd.DataFrame,
    weights: dict = None,
) -> dict:
    """
    Calculate the composite Team Strength Index.

    Args:
        team_batting: Batting stats for team players
        team_bowling: Bowling stats for team players
        weights: Custom weights for each component (must sum to 1.0)

    Returns:
        Dictionary with overall index and component scores
    """
    if weights is None:
        weights = {
            "batting_depth": 0.25,
            "bowling_strength": 0.25,
            "death_bowling": 0.20,
            "powerplay": 0.15,
            "bench_strength": 0.15,
        }

    # Calculate each component
    batting_depth = calculate_team_batting_depth(team_batting)
    bowling_strength = calculate_team_bowling_strength(team_bowling)
    death_rating = calculate_death_bowling_rating(team_bowling)
    powerplay_rating = calculate_powerplay_rating(team_batting, team_bowling)

    # Bench strength: how deep is the squad beyond starting XI
    total_players = len(team_batting)
    bench_strength = min(total_players / 20, 1.0) * 100  # 20+ players = full bench

    # Composite index
    index = (
        batting_depth * weights["batting_depth"] +
        bowling_strength * weights["bowling_strength"] +
        death_rating * weights["death_bowling"] +
        powerplay_rating * weights["powerplay"] +
        bench_strength * weights["bench_strength"]
    )

    return {
        "team_strength_index": round(index, 2),
        "batting_depth_score": round(batting_depth, 2),
        "bowling_strength_score": round(bowling_strength, 2),
        "death_bowling_score": round(death_rating, 2),
        "powerplay_score": round(powerplay_rating, 2),
        "bench_strength_score": round(bench_strength, 2),
    }


def calculate_all_team_strengths(
    batting_stats: pd.DataFrame,
    bowling_stats: pd.DataFrame,
    team_squads: dict,
) -> pd.DataFrame:
    """
    Calculate Team Strength Index for ALL teams.

    Args:
        batting_stats: Full batting stats DataFrame
        bowling_stats: Full bowling stats DataFrame
        team_squads: Dict mapping team name -> list of player names

    Returns:
        DataFrame with one row per team
    """
    print("  Calculating team strength indices...")
    records = []

    for team, players in team_squads.items():
        team_bat = batting_stats[batting_stats["batter"].isin(players)]
        team_bowl = bowling_stats[bowling_stats["bowler"].isin(players)]

        strength = calculate_team_strength_index(team_bat, team_bowl)
        strength["team"] = team
        records.append(strength)

    df = pd.DataFrame(records)
    df = df.sort_values("team_strength_index", ascending=False).reset_index(drop=True)
    print(f"    Calculated strength for {len(df)} teams")
    return df
