"""
Master feature builder — combines ALL features into a single match-level dataset.

This is the final feature engineering step. It takes every match and
creates a feature-rich row that the ML model can learn from.

HOW TO RUN:
    python -m src.features.build_match_features

WHAT IT DOES:
    1. Loads matches, deliveries, player stats, venue stats
    2. For each match, generates 80-150 features by calling the individual
       feature modules (player_form, team_strength, head_to_head, venue)
    3. Saves the complete feature table to data/processed/match_features.csv

BEGINNER NOTES:
    - This script "orchestrates" all the feature modules
    - The output is the training data for your ML model
    - Each row = one match, each column = one feature
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODEL_CONFIG

from src.features.player_form import calculate_batting_form, calculate_bowling_form
from src.features.head_to_head import calculate_team_h2h
from src.features.venue_features import calculate_venue_stats, calculate_team_venue_record


def compute_team_recent_form(matches: pd.DataFrame, team: str, before_date, window: int = 5) -> dict:
    """
    Calculate a team's recent form before a specific date.

    This answers: "How has this team been performing in their last N matches?"
    """
    team_matches = matches[
        ((matches["team1"] == team) | (matches["team2"] == team)) &
        (matches["date"] < before_date)
    ].sort_values("date", ascending=False).head(window)

    total = len(team_matches)
    if total == 0:
        return {
            "recent_wins": 0, "recent_losses": 0, "recent_win_pct": 0.5,
            "recent_avg_margin": 0, "recent_matches": 0,
        }

    wins = len(team_matches[team_matches["winner"] == team])
    losses = total - wins

    # Average margin of wins
    win_matches = team_matches[team_matches["winner"] == team]
    avg_margin = 0
    if len(win_matches) > 0:
        avg_margin = (
            win_matches["win_by_runs"].mean() + win_matches["win_by_wickets"].mean() * 10
        )

    return {
        "recent_wins": wins,
        "recent_losses": losses,
        "recent_win_pct": round(wins / total, 4),
        "recent_avg_margin": round(avg_margin, 2),
        "recent_matches": total,
    }


def compute_season_standings(matches: pd.DataFrame, team: str, season: str) -> dict:
    """Calculate team's current season standings (wins, NRR approximation, position)."""
    season_matches = matches[
        (matches["season"] == season) &
        ((matches["team1"] == team) | (matches["team2"] == team))
    ]

    total = len(season_matches)
    wins = len(season_matches[season_matches["winner"] == team])
    points = wins * 2  # IPL: 2 points per win

    return {
        "season_matches_played": total,
        "season_wins": wins,
        "season_losses": total - wins,
        "season_points": points,
        "season_win_pct": round(wins / max(total, 1), 4),
    }


def build_features_for_match(
    match: pd.Series,
    matches_df: pd.DataFrame,
    deliveries_df: pd.DataFrame,
    venue_stats: pd.DataFrame,
) -> dict:
    """
    Build the complete feature vector for a single match.

    This function is called for EVERY match in the dataset.
    It combines all the different feature types into one row.
    """
    features = {}
    team1 = match["team1"]
    team2 = match["team2"]
    venue = match["venue"]
    match_date = match["date"]
    season = match["season"]

    # ── 1. Team Recent Form ──
    form1 = compute_team_recent_form(matches_df, team1, match_date, window=5)
    form2 = compute_team_recent_form(matches_df, team2, match_date, window=5)

    for k, v in form1.items():
        features[f"team1_{k}"] = v
    for k, v in form2.items():
        features[f"team2_{k}"] = v

    # Form difference (team1 advantage)
    features["form_win_pct_diff"] = form1["recent_win_pct"] - form2["recent_win_pct"]

    # ── 2. Season Standings ──
    stand1 = compute_season_standings(matches_df, team1, season)
    stand2 = compute_season_standings(matches_df, team2, season)

    for k, v in stand1.items():
        features[f"team1_{k}"] = v
    for k, v in stand2.items():
        features[f"team2_{k}"] = v

    features["points_diff"] = stand1["season_points"] - stand2["season_points"]

    # ── 3. Head-to-Head ──
    h2h = calculate_team_h2h(matches_df[matches_df["date"] < match_date], team1, team2)
    for k, v in h2h.items():
        features[k] = v

    # ── 4. Venue Features ──
    venue_row = venue_stats[venue_stats["venue"] == venue]
    if len(venue_row) > 0:
        venue_row = venue_row.iloc[0]
        for col in venue_row.index:
            if col not in ["venue", "city"]:
                features[col] = venue_row[col]

    # Team1 record at this venue
    tv1 = calculate_team_venue_record(
        matches_df[matches_df["date"] < match_date], team1, venue
    )
    tv2 = calculate_team_venue_record(
        matches_df[matches_df["date"] < match_date], team2, venue
    )
    features["team1_venue_win_pct"] = tv1["team_venue_win_pct"]
    features["team2_venue_win_pct"] = tv2["team_venue_win_pct"]
    features["venue_win_pct_diff"] = tv1["team_venue_win_pct"] - tv2["team_venue_win_pct"]

    # ── 5. Toss Features ──
    features["toss_winner_is_team1"] = 1 if match.get("toss_winner") == team1 else 0
    features["toss_decision_bat"] = 1 if match.get("toss_decision") == "bat" else 0

    # ── 6. Key Player Form (top 3 batters + top 2 bowlers per team) ──
    # This uses the deliveries before this match date
    past_deliveries = deliveries_df[
        deliveries_df["match_id"].isin(
            matches_df[matches_df["date"] < match_date]["match_id"]
        )
    ]

    # Get team's players from recent matches
    for team_label, team_name in [("team1", team1), ("team2", team2)]:
        team_recent_deliveries = past_deliveries[
            past_deliveries["batting_team"] == team_name
        ]
        team_batters = team_recent_deliveries["batter"].value_counts().head(5).index.tolist()

        bat_forms = []
        for batter in team_batters[:3]:
            bf = calculate_batting_form(past_deliveries, batter, window=10)
            bat_forms.append(bf["form_index"])

        features[f"{team_label}_top_batter_form_avg"] = round(np.mean(bat_forms), 2) if bat_forms else 0

        team_bowler_deliveries = past_deliveries[
            past_deliveries["batting_team"] != team_name
        ]
        # Bowlers from this team
        team_prev_matches = matches_df[
            (matches_df["date"] < match_date) &
            ((matches_df["team1"] == team_name) | (matches_df["team2"] == team_name))
        ]
        team_match_ids = team_prev_matches["match_id"].tolist()
        team_bowling_data = past_deliveries[
            (past_deliveries["match_id"].isin(team_match_ids)) &
            (past_deliveries["batting_team"] != team_name)
        ]
        team_bowlers = team_bowling_data["bowler"].value_counts().head(4).index.tolist()

        bowl_forms = []
        for bowler in team_bowlers[:2]:
            bf = calculate_bowling_form(past_deliveries, bowler, window=10)
            bowl_forms.append(bf["form_index"])

        features[f"{team_label}_top_bowler_form_avg"] = round(np.mean(bowl_forms), 2) if bowl_forms else 0

    # ── 7. Match Context ──
    features["match_number_in_season"] = stand1["season_matches_played"] + stand2["season_matches_played"]

    # ── 8. Target variable ──
    features["team1_won"] = 1 if match.get("winner") == team1 else 0
    features["winner"] = match.get("winner", "")

    # ── Metadata (excluded from model features) ──
    features["match_id"] = match["match_id"]
    features["date"] = match_date
    features["team1"] = team1
    features["team2"] = team2

    return features


def build_all_match_features():
    """
    Build the complete feature-engineered dataset for ALL matches.

    This is the main function that creates the training dataset.
    """
    print("\n" + "#" * 60)
    print("#  BUILDING MATCH FEATURES")
    print("#  This creates the training data for your ML model")
    print("#" * 60)

    # Load data
    matches_path = PROCESSED_DATA_DIR / "matches.csv"
    deliveries_path = PROCESSED_DATA_DIR / "deliveries.csv"

    if not matches_path.exists() or not deliveries_path.exists():
        print("ERROR: Data files not found. Run the data collection scripts first!")
        print("  1. python -m src.data_collection.download_cricsheet")
        print("  2. python -m src.data_collection.parse_matches")
        return None

    print("\nLoading data...")
    matches = pd.read_csv(matches_path, parse_dates=["date"])
    deliveries = pd.read_csv(deliveries_path)
    print(f"  Loaded {len(matches)} matches, {len(deliveries):,} deliveries")

    # Pre-compute venue stats (so we don't recalculate per match)
    print("\nPre-computing venue statistics...")
    venue_stats = calculate_venue_stats(matches, deliveries)

    # Build features for each match
    print(f"\nBuilding features for {len(matches)} matches...")
    print("  (This may take a few minutes for large datasets)")

    all_features = []
    for idx, match in matches.iterrows():
        if idx % 100 == 0:
            print(f"  Processing match {idx + 1}/{len(matches)}...")

        # Skip matches with no result
        if not match.get("winner") or match["winner"] in ["no result", "tie"]:
            continue

        features = build_features_for_match(match, matches, deliveries, venue_stats)
        all_features.append(features)

    # Create final DataFrame
    features_df = pd.DataFrame(all_features)

    # Save
    output_path = PROCESSED_DATA_DIR / "match_features.csv"
    features_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)
    print(f"  Matches with features: {len(features_df)}")
    print(f"  Features per match:    {len(features_df.columns)}")
    print(f"  Saved to: {output_path}")
    print(f"\nNext step: Run  python -m src.models.train_model")

    return features_df


if __name__ == "__main__":
    build_all_match_features()
