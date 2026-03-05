"""
Parse downloaded Cricsheet JSON files into clean DataFrames.

This script reads the raw JSON match files and creates two clean CSV files:
1. matches.csv - One row per match with match-level information
2. deliveries.csv - One row per ball (delivery) with ball-by-ball detail

HOW TO RUN:
    python -m src.data_collection.parse_matches

WHAT IT DOES:
    1. Reads every JSON file in data/raw/ipl_matches/
    2. Extracts match info (teams, venue, winner, toss, etc.)
    3. Extracts ball-by-ball data (runs, wickets, extras, etc.)
    4. Saves clean CSV files to data/processed/

BEGINNER NOTES:
    - JSON (JavaScript Object Notation) is a common data format, like a nested dictionary
    - We use try/except blocks to handle files that might be corrupted or have missing fields
    - .get() is a safe way to access dictionary keys - returns a default if the key doesn't exist
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def parse_match_info(filepath: str) -> dict:
    """
    Extract match-level information from a Cricsheet JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with match info, or None if parsing fails

    BEGINNER NOTE:
        info.get("key", default_value) tries to get "key" from the dictionary.
        If the key doesn't exist, it returns default_value instead of crashing.
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    info = data.get("info", {})
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})
    teams = info.get("teams", [])
    dates = info.get("dates", [])
    registry = info.get("registry", {}).get("people", {})

    # Determine winner
    winner = outcome.get("winner", "")

    # Check if match was abandoned or no result
    if not winner:
        result_type = outcome.get("result", "")
        if result_type in ["no result", "tie"]:
            winner = result_type

    # Get player of match
    pom_list = info.get("player_of_match", [])
    player_of_match = pom_list[0] if pom_list else ""

    # Get match type and event info
    event = info.get("event", {})

    match_record = {
        "match_id": Path(filepath).stem,  # filename without .json
        "season": str(info.get("season", "")),
        "date": dates[0] if dates else "",
        "team1": teams[0] if len(teams) > 0 else "",
        "team2": teams[1] if len(teams) > 1 else "",
        "venue": info.get("venue", ""),
        "city": info.get("city", ""),
        "toss_winner": toss.get("winner", ""),
        "toss_decision": toss.get("decision", ""),
        "winner": winner,
        "win_by_runs": outcome.get("by", {}).get("runs", 0),
        "win_by_wickets": outcome.get("by", {}).get("wickets", 0),
        "player_of_match": player_of_match,
        "match_type": info.get("match_type", "T20"),
        "gender": info.get("gender", "male"),
        "event_name": event.get("name", ""),
        "event_match_number": event.get("match_number", ""),
        "overs_per_side": info.get("overs", 20),
    }

    return match_record


def parse_deliveries(filepath: str) -> list:
    """
    Extract ball-by-ball delivery data from a Cricsheet JSON file.

    Returns a list of dictionaries, one per delivery (ball bowled).

    BEGINNER NOTE:
        Cricket terminology:
        - "innings" = one team's turn to bat (each match has 2 innings)
        - "over" = a set of 6 balls bowled by one bowler
        - "delivery" = a single ball bowled
        - "extras" = runs not credited to the batter (wides, no-balls, byes, leg-byes)
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    match_id = Path(filepath).stem
    innings_data = data.get("innings", [])
    deliveries = []

    for innings_num, innings in enumerate(innings_data, 1):
        batting_team = innings.get("team", "")
        overs = innings.get("overs", [])

        for over_data in overs:
            over_num = over_data.get("over", 0)

            for ball_num, delivery in enumerate(over_data.get("deliveries", []), 1):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                delivery_record = {
                    "match_id": match_id,
                    "innings": innings_num,
                    "batting_team": batting_team,
                    "over": over_num,
                    "ball": ball_num,
                    "batter": delivery.get("batter", ""),
                    "bowler": delivery.get("bowler", ""),
                    "non_striker": delivery.get("non_striker", ""),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "extras_wides": extras.get("wides", 0),
                    "extras_noballs": extras.get("noballs", 0),
                    "extras_byes": extras.get("byes", 0),
                    "extras_legbyes": extras.get("legbyes", 0),
                    "extras_penalty": extras.get("penalty", 0),
                    "is_wicket": 1 if wickets else 0,
                    "wicket_kind": wickets[0].get("kind", "") if wickets else "",
                    "wicket_player": wickets[0].get("player_out", "") if wickets else "",
                    "wicket_fielders": ", ".join(
                        [f.get("name", "") for f in wickets[0].get("fielders", [])]
                    ) if wickets else "",
                }

                # Determine phase of innings
                if over_num < 6:
                    delivery_record["phase"] = "powerplay"
                elif over_num < 15:
                    delivery_record["phase"] = "middle"
                else:
                    delivery_record["phase"] = "death"

                deliveries.append(delivery_record)

    return deliveries


def parse_all_ipl_matches():
    """
    Parse ALL IPL match JSON files into clean CSV files.

    Creates two files:
        - data/processed/matches.csv (match-level summary)
        - data/processed/deliveries.csv (ball-by-ball detail)
    """
    match_dir = RAW_DATA_DIR / "ipl_matches"

    if not match_dir.exists():
        print(f"ERROR: {match_dir} not found. Run download_cricsheet.py first!")
        print("  Command: python -m src.data_collection.download_cricsheet")
        return None, None

    # Find all JSON files
    json_files = sorted(match_dir.glob("*.json"))
    print(f"\nFound {len(json_files)} match files to parse...")

    if len(json_files) == 0:
        # Check subdirectories
        json_files = sorted(match_dir.rglob("*.json"))
        print(f"Found {len(json_files)} files in subdirectories")

    # Parse each match
    all_matches = []
    all_deliveries = []

    for filepath in tqdm(json_files, desc="Parsing matches"):
        # Parse match info
        match_info = parse_match_info(str(filepath))
        if match_info:
            all_matches.append(match_info)

        # Parse ball-by-ball data
        delivery_data = parse_deliveries(str(filepath))
        all_deliveries.extend(delivery_data)

    # Create DataFrames
    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)

    # Sort by date
    if "date" in matches_df.columns and len(matches_df) > 0:
        matches_df["date"] = pd.to_datetime(matches_df["date"], errors="coerce")
        matches_df = matches_df.sort_values("date").reset_index(drop=True)

    # Save to CSV
    matches_path = PROCESSED_DATA_DIR / "matches.csv"
    deliveries_path = PROCESSED_DATA_DIR / "deliveries.csv"

    matches_df.to_csv(matches_path, index=False)
    deliveries_df.to_csv(deliveries_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("PARSING COMPLETE!")
    print("=" * 60)
    print(f"  Matches parsed:    {len(matches_df)}")
    print(f"  Deliveries parsed: {len(deliveries_df):,}")

    if len(matches_df) > 0:
        print(f"\n  Seasons covered:   {matches_df['season'].nunique()}")
        print(f"  Date range:        {matches_df['date'].min()} to {matches_df['date'].max()}")
        print(f"  Teams:             {matches_df['team1'].nunique()}")
        print(f"  Venues:            {matches_df['venue'].nunique()}")

    print(f"\n  Saved to:")
    print(f"    {matches_path}")
    print(f"    {deliveries_path}")
    print(f"\nNext step: Run  python -m src.data_collection.build_player_stats")

    return matches_df, deliveries_df


if __name__ == "__main__":
    parse_all_ipl_matches()
