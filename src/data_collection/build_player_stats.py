"""
Build comprehensive player statistics from ball-by-ball delivery data.

This script processes the deliveries.csv file to create detailed
batting and bowling statistics for every player who has played in the IPL.

HOW TO RUN:
    python -m src.data_collection.build_player_stats

WHAT IT DOES:
    1. Reads deliveries.csv (ball-by-ball data)
    2. Aggregates batting stats: runs, balls faced, strike rate, boundaries, etc.
    3. Aggregates bowling stats: overs, wickets, economy, dot balls, etc.
    4. Calculates phase-specific stats (powerplay, middle, death)
    5. Saves player_batting_stats.csv and player_bowling_stats.csv

BEGINNER NOTES:
    - groupby() groups rows by a column value, like a pivot table in Excel
    - agg() applies aggregate functions (sum, mean, count) to grouped data
    - lambda is a one-line function, e.g., lambda x: x.sum() means "sum all values"
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_DATA_DIR


def build_batting_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate batting statistics for every batter.

    Stats include: matches, innings, runs, balls faced, strike rate,
    boundaries (4s & 6s), dot ball %, 50s, 100s, and phase-wise breakdown.
    """
    print("  Building batting stats...")

    # ── Overall batting stats ──
    batting = deliveries.groupby("batter").agg(
        matches=("match_id", "nunique"),
        innings=("match_id", lambda x: x.nunique()),  # Simplified; same as matches for now
        total_runs=("runs_batter", "sum"),
        balls_faced=("runs_batter", "count"),  # Every delivery faced
        fours=("runs_batter", lambda x: (x == 4).sum()),
        sixes=("runs_batter", lambda x: (x == 6).sum()),
        dot_balls=("runs_batter", lambda x: (x == 0).sum()),
        ones=("runs_batter", lambda x: (x == 1).sum()),
        twos=("runs_batter", lambda x: (x == 2).sum()),
        threes=("runs_batter", lambda x: (x == 3).sum()),
    ).reset_index()

    # Calculate derived metrics
    batting["strike_rate"] = np.where(
        batting["balls_faced"] > 0,
        (batting["total_runs"] / batting["balls_faced"]) * 100,
        0
    )
    batting["batting_average"] = np.where(
        batting["innings"] > 0,
        batting["total_runs"] / batting["innings"],
        0
    )
    batting["boundary_pct"] = np.where(
        batting["balls_faced"] > 0,
        ((batting["fours"] + batting["sixes"]) / batting["balls_faced"]) * 100,
        0
    )
    batting["dot_ball_pct"] = np.where(
        batting["balls_faced"] > 0,
        (batting["dot_balls"] / batting["balls_faced"]) * 100,
        0
    )

    # ── Per-innings scores (for 50s, 100s, highest score) ──
    innings_scores = deliveries.groupby(["match_id", "innings", "batter"]).agg(
        innings_runs=("runs_batter", "sum"),
        innings_balls=("runs_batter", "count"),
    ).reset_index()

    fifties = innings_scores[
        (innings_scores["innings_runs"] >= 50) & (innings_scores["innings_runs"] < 100)
    ].groupby("batter").size().reset_index(name="fifties")

    hundreds = innings_scores[
        innings_scores["innings_runs"] >= 100
    ].groupby("batter").size().reset_index(name="hundreds")

    highest = innings_scores.groupby("batter")["innings_runs"].max().reset_index(name="highest_score")

    batting = batting.merge(fifties, on="batter", how="left")
    batting = batting.merge(hundreds, on="batter", how="left")
    batting = batting.merge(highest, on="batter", how="left")
    batting[["fifties", "hundreds"]] = batting[["fifties", "hundreds"]].fillna(0).astype(int)

    # ── Phase-wise batting stats ──
    for phase in ["powerplay", "middle", "death"]:
        phase_data = deliveries[deliveries["phase"] == phase]
        phase_stats = phase_data.groupby("batter").agg(
            **{
                f"{phase}_runs": ("runs_batter", "sum"),
                f"{phase}_balls": ("runs_batter", "count"),
                f"{phase}_fours": ("runs_batter", lambda x: (x == 4).sum()),
                f"{phase}_sixes": ("runs_batter", lambda x: (x == 6).sum()),
            }
        ).reset_index()

        phase_stats[f"{phase}_sr"] = np.where(
            phase_stats[f"{phase}_balls"] > 0,
            (phase_stats[f"{phase}_runs"] / phase_stats[f"{phase}_balls"]) * 100,
            0
        )
        batting = batting.merge(phase_stats, on="batter", how="left")

    # Round numeric columns
    numeric_cols = batting.select_dtypes(include=[np.number]).columns
    batting[numeric_cols] = batting[numeric_cols].round(2)
    batting = batting.fillna(0)

    print(f"    Batting stats for {len(batting)} players")
    return batting


def build_bowling_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate bowling statistics for every bowler.

    Stats include: matches, overs, wickets, economy, average,
    strike rate, dot ball %, and phase-wise breakdown.
    """
    print("  Building bowling stats...")

    # Filter out wides and no-balls for "legal deliveries" count
    legal_deliveries = deliveries[
        (deliveries["extras_wides"] == 0) & (deliveries["extras_noballs"] == 0)
    ]

    # ── Overall bowling stats ──
    bowling = deliveries.groupby("bowler").agg(
        matches=("match_id", "nunique"),
        total_balls=("bowler", "count"),
        runs_conceded=("runs_total", "sum"),
        wickets=("is_wicket", "sum"),
        wides=("extras_wides", lambda x: (x > 0).sum()),
        noballs=("extras_noballs", lambda x: (x > 0).sum()),
        dot_balls=("runs_total", lambda x: (x == 0).sum()),
        fours_conceded=("runs_batter", lambda x: (x == 4).sum()),
        sixes_conceded=("runs_batter", lambda x: (x == 6).sum()),
    ).reset_index()

    # Legal deliveries count (for overs calculation)
    legal_counts = legal_deliveries.groupby("bowler").size().reset_index(name="legal_balls")
    bowling = bowling.merge(legal_counts, on="bowler", how="left")
    bowling["legal_balls"] = bowling["legal_balls"].fillna(0)

    # Calculate overs (6 legal deliveries = 1 over)
    bowling["overs"] = bowling["legal_balls"] // 6 + (bowling["legal_balls"] % 6) / 10

    # Economy rate = runs per over
    bowling["economy"] = np.where(
        bowling["overs"] > 0,
        bowling["runs_conceded"] / (bowling["legal_balls"] / 6),
        0
    )

    # Bowling average = runs per wicket
    bowling["bowling_average"] = np.where(
        bowling["wickets"] > 0,
        bowling["runs_conceded"] / bowling["wickets"],
        999  # Very high = bad (no wickets taken)
    )

    # Bowling strike rate = balls per wicket
    bowling["bowling_strike_rate"] = np.where(
        bowling["wickets"] > 0,
        bowling["legal_balls"] / bowling["wickets"],
        999
    )

    # Dot ball percentage
    bowling["dot_ball_pct"] = np.where(
        bowling["total_balls"] > 0,
        (bowling["dot_balls"] / bowling["total_balls"]) * 100,
        0
    )

    # ── Phase-wise bowling stats ──
    for phase in ["powerplay", "middle", "death"]:
        phase_data = deliveries[deliveries["phase"] == phase]
        phase_legal = phase_data[
            (phase_data["extras_wides"] == 0) & (phase_data["extras_noballs"] == 0)
        ]

        phase_stats = phase_data.groupby("bowler").agg(
            **{
                f"{phase}_runs_conceded": ("runs_total", "sum"),
                f"{phase}_wickets": ("is_wicket", "sum"),
                f"{phase}_balls": ("bowler", "count"),
            }
        ).reset_index()

        phase_legal_counts = phase_legal.groupby("bowler").size().reset_index(
            name=f"{phase}_legal_balls"
        )
        phase_stats = phase_stats.merge(phase_legal_counts, on="bowler", how="left")
        phase_stats[f"{phase}_legal_balls"] = phase_stats[f"{phase}_legal_balls"].fillna(0)

        phase_stats[f"{phase}_economy"] = np.where(
            phase_stats[f"{phase}_legal_balls"] > 0,
            phase_stats[f"{phase}_runs_conceded"] / (phase_stats[f"{phase}_legal_balls"] / 6),
            0
        )

        bowling = bowling.merge(phase_stats, on="bowler", how="left")

    # ── Per-match bowling (for best figures) ──
    match_bowling = deliveries.groupby(["match_id", "bowler"]).agg(
        match_wickets=("is_wicket", "sum"),
        match_runs=("runs_total", "sum"),
    ).reset_index()

    best_figures = match_bowling.loc[
        match_bowling.groupby("bowler")["match_wickets"].idxmax()
    ][["bowler", "match_wickets", "match_runs"]].rename(columns={
        "match_wickets": "best_wickets",
        "match_runs": "best_runs_in_best"
    })

    bowling = bowling.merge(best_figures, on="bowler", how="left")

    # Round and clean
    numeric_cols = bowling.select_dtypes(include=[np.number]).columns
    bowling[numeric_cols] = bowling[numeric_cols].round(2)
    bowling = bowling.fillna(0)

    print(f"    Bowling stats for {len(bowling)} players")
    return bowling


def build_all_player_stats():
    """Build and save all player statistics."""
    print("\n" + "#" * 60)
    print("#  BUILDING PLAYER STATISTICS")
    print("#" * 60)

    deliveries_path = PROCESSED_DATA_DIR / "deliveries.csv"
    if not deliveries_path.exists():
        print(f"ERROR: {deliveries_path} not found. Run parse_matches.py first!")
        return

    print(f"\nLoading deliveries from {deliveries_path}...")
    deliveries = pd.read_csv(deliveries_path)
    print(f"  Loaded {len(deliveries):,} deliveries")

    # Build stats
    batting_stats = build_batting_stats(deliveries)
    bowling_stats = build_bowling_stats(deliveries)

    # Save
    batting_path = PROCESSED_DATA_DIR / "player_batting_stats.csv"
    bowling_path = PROCESSED_DATA_DIR / "player_bowling_stats.csv"

    batting_stats.to_csv(batting_path, index=False)
    bowling_stats.to_csv(bowling_path, index=False)

    print("\n" + "=" * 60)
    print("PLAYER STATS COMPLETE!")
    print("=" * 60)
    print(f"  Batters:  {len(batting_stats)} players")
    print(f"  Bowlers:  {len(bowling_stats)} players")
    print(f"\n  Saved to:")
    print(f"    {batting_path}")
    print(f"    {bowling_path}")
    print(f"\nNext step: Run  python -m src.data_collection.weather_collector")


if __name__ == "__main__":
    build_all_player_stats()
