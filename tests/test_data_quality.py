"""
Data quality validation checks.

Run these after every data pipeline update to ensure
the data hasn't been corrupted or is missing critical fields.

HOW TO RUN:
    python -m tests.test_data_quality

BEGINNER NOTES:
    - Data quality is the foundation of any ML project
    - "Garbage in, garbage out" — bad data = bad predictions
    - These checks catch common problems: missing values, duplicates,
      impossible values (e.g., strike rate > 700), and format issues
"""

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PROCESSED_DATA_DIR


def check_matches_quality():
    """Validate the matches.csv file."""
    print("\n--- Checking matches.csv ---")
    path = PROCESSED_DATA_DIR / "matches.csv"

    if not path.exists():
        print("  SKIP: File not found")
        return

    df = pd.read_csv(path)
    issues = []

    # Check for duplicates
    dupes = df["match_id"].duplicated().sum()
    if dupes > 0:
        issues.append(f"  WARN: {dupes} duplicate match IDs found")

    # Check required columns exist
    required_cols = ["match_id", "season", "date", "team1", "team2", "venue", "winner"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"  ERROR: Missing columns: {missing_cols}")

    # Check for null values in critical columns
    for col in ["match_id", "team1", "team2", "date"]:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                issues.append(f"  WARN: {nulls} null values in '{col}'")

    # Check date range is reasonable
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        min_date = df["date"].min()
        max_date = df["date"].max()
        if min_date and min_date.year < 2007:
            issues.append(f"  WARN: Suspicious early date: {min_date}")

    # Summary
    print(f"  Rows: {len(df)}")
    print(f"  Seasons: {df['season'].nunique() if 'season' in df.columns else 'N/A'}")
    print(f"  Date range: {min_date} to {max_date}" if "date" in df.columns else "")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  PASSED: All quality checks passed")


def check_deliveries_quality():
    """Validate the deliveries.csv file."""
    print("\n--- Checking deliveries.csv ---")
    path = PROCESSED_DATA_DIR / "deliveries.csv"

    if not path.exists():
        print("  SKIP: File not found")
        return

    df = pd.read_csv(path)
    issues = []

    # Check runs are reasonable
    if "runs_batter" in df.columns:
        max_runs = df["runs_batter"].max()
        if max_runs > 6:
            issues.append(f"  WARN: Max batter runs = {max_runs} (should be 0-6)")

    # Check overs are reasonable
    if "over" in df.columns:
        max_over = df["over"].max()
        if max_over > 20:
            issues.append(f"  WARN: Max over = {max_over} (should be 0-19 or 0-20)")

    # Check for missing player names
    for col in ["batter", "bowler"]:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            empty = (df[col] == "").sum()
            if nulls + empty > 0:
                issues.append(f"  WARN: {nulls + empty} missing {col} names")

    # Check innings values
    if "innings" in df.columns:
        unique_innings = df["innings"].unique()
        invalid = [i for i in unique_innings if i not in [1, 2, 3, 4]]  # Super over = 3/4
        if invalid:
            issues.append(f"  WARN: Unexpected innings values: {invalid}")

    print(f"  Rows: {len(df):,}")
    print(f"  Unique matches: {df['match_id'].nunique()}")
    print(f"  Unique batters: {df['batter'].nunique()}")
    print(f"  Unique bowlers: {df['bowler'].nunique()}")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  PASSED: All quality checks passed")


def check_player_stats_quality():
    """Validate player stats files."""
    for stat_type in ["batting", "bowling"]:
        print(f"\n--- Checking player_{stat_type}_stats.csv ---")
        path = PROCESSED_DATA_DIR / f"player_{stat_type}_stats.csv"

        if not path.exists():
            print("  SKIP: File not found")
            continue

        df = pd.read_csv(path)
        issues = []

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                issues.append(f"  WARN: {inf_count} infinite values in '{col}'")

        # Check for negative values where they shouldn't be
        non_negative_cols = ["matches", "total_runs", "balls_faced", "wickets", "fours", "sixes"]
        for col in non_negative_cols:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    issues.append(f"  ERROR: {neg} negative values in '{col}'")

        # Check strike rate range
        if "strike_rate" in df.columns:
            extreme_sr = (df["strike_rate"] > 400).sum()
            if extreme_sr > 0:
                issues.append(f"  WARN: {extreme_sr} players with SR > 400 (may be low sample)")

        print(f"  Players: {len(df)}")
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  PASSED: All quality checks passed")


def check_feature_quality():
    """Validate the feature-engineered dataset."""
    print("\n--- Checking match_features.csv ---")
    path = PROCESSED_DATA_DIR / "match_features.csv"

    if not path.exists():
        print("  SKIP: File not found")
        return

    df = pd.read_csv(path)
    issues = []

    # Check for NaN in target variable
    if "team1_won" in df.columns:
        nulls = df["team1_won"].isnull().sum()
        if nulls > 0:
            issues.append(f"  ERROR: {nulls} null values in target variable 'team1_won'")

    # Check for excessive NaN in features
    null_pct = df.isnull().mean()
    high_null_cols = null_pct[null_pct > 0.3].index.tolist()
    if high_null_cols:
        issues.append(f"  WARN: {len(high_null_cols)} features have >30% null values")

    # Check for constant features (no variance = useless)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].std() == 0:
            issues.append(f"  WARN: '{col}' has zero variance (constant feature)")

    print(f"  Matches: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Overall null rate: {df.isnull().mean().mean():.1%}")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  PASSED: All quality checks passed")


def run_all_checks():
    """Run all data quality checks."""
    print("\n" + "#" * 60)
    print("#  DATA QUALITY VALIDATION")
    print("#" * 60)

    check_matches_quality()
    check_deliveries_quality()
    check_player_stats_quality()
    check_feature_quality()

    print("\n" + "=" * 60)
    print("Data quality check complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
