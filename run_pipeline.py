"""
IPL Prediction Agent — Master Pipeline Runner

This script runs the entire pipeline from data download to trained model.
Perfect for first-time setup or full refresh.

HOW TO RUN:
    python run_pipeline.py

WHAT IT DOES (in order):
    1. Downloads IPL match data from Cricsheet
    2. Parses JSON files into clean CSVs
    3. Builds player batting and bowling statistics
    4. Collects weather data for venues
    5. Engineers all match features
    6. Trains the ensemble prediction model
    7. Runs back-testing to validate accuracy
    8. Runs data quality checks

ESTIMATED TIME: 15-30 minutes (depending on internet speed and data size)

BEGINNER NOTES:
    Run this once to set everything up. After that, use individual
    scripts for specific updates (e.g., just retrain the model).
"""

import sys
import time
from datetime import datetime


def run_step(step_num, description, module_path):
    """Run a pipeline step and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

    start = time.time()

    try:
        # Dynamically import and run the module's main function
        parts = module_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[parts[-1]])

        # Each module has a main function - find and call it
        main_funcs = {
            "src.data_collection.download_cricsheet": "download_all",
            "src.data_collection.parse_matches": "parse_all_ipl_matches",
            "src.data_collection.build_player_stats": "build_all_player_stats",
            "src.data_collection.weather_collector": "collect_all_venue_weather",
            "src.features.build_match_features": "build_all_match_features",
            "src.models.train_model": "train_full_pipeline",
            "src.models.backtest": "run_full_backtest",
            "tests.test_data_quality": "run_all_checks",
        }

        func_name = main_funcs.get(module_path)
        if func_name and hasattr(module, func_name):
            getattr(module, func_name)()
        else:
            print(f"  Warning: Could not find main function for {module_path}")

        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f} seconds")
        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ERROR after {elapsed:.1f}s: {e}")
        print(f"  You can run this step manually: python -m {module_path}")
        return False


def main():
    """Run the complete IPL Prediction Agent pipeline."""
    print("#" * 60)
    print("#")
    print("#   IPL PREDICTION AI AGENT")
    print("#   Full Pipeline Runner")
    print("#")
    print(f"#   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#")
    print("#" * 60)

    pipeline_start = time.time()
    results = {}

    # Define the pipeline steps
    steps = [
        (1, "Download Cricket Data", "src.data_collection.download_cricsheet"),
        (2, "Parse Match Files", "src.data_collection.parse_matches"),
        (3, "Build Player Statistics", "src.data_collection.build_player_stats"),
        (4, "Collect Weather Data", "src.data_collection.weather_collector"),
        (5, "Engineer Match Features", "src.features.build_match_features"),
        (6, "Train Prediction Model", "src.models.train_model"),
        (7, "Run Back-Testing", "src.models.backtest"),
        (8, "Run Data Quality Checks", "tests.test_data_quality"),
    ]

    for step_num, description, module_path in steps:
        success = run_step(step_num, description, module_path)
        results[description] = "PASSED" if success else "FAILED"

    # Final summary
    total_time = time.time() - pipeline_start
    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE!")
    print("#" * 60)
    print(f"\n  Total time: {total_time / 60:.1f} minutes")
    print(f"\n  Results:")
    for step_name, status in results.items():
        emoji = "  " if status == "PASSED" else "  "
        print(f"    {emoji} {step_name}: {status}")

    print(f"\n  Next steps:")
    print(f"    1. Launch dashboard: streamlit run src/dashboard/app.py")
    print(f"    2. Make a prediction: python -m src.models.predict --team1 'CSK' --team2 'MI'")
    print(f"    3. Predict tournament: python -m src.models.predict --tournament")


if __name__ == "__main__":
    main()
