"""
Back-testing framework — validate the model on historical seasons.

Back-testing replays past IPL seasons match-by-match, as if you were
predicting live. This is the most honest way to evaluate your model.

HOW TO RUN:
    python -m src.models.backtest

WHAT IT DOES:
    1. Takes the last N seasons of IPL data
    2. For each season, trains the model on all data BEFORE that season
    3. Predicts each match in the season one at a time
    4. After each match, optionally updates the model with the result
    5. Measures accuracy, calibration, and Brier score per season

WHY THIS MATTERS:
    - Regular train/test split can be misleading for time-series data
    - Back-testing simulates real-world usage: you only know the past
    - It also tests the adaptive learning: does the model improve mid-season?
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_DATA_DIR, MODELS_DIR


EXCLUDE_COLS = [
    "match_id", "date", "team1", "team2", "winner", "team1_won",
    "season", "venue", "city",
]


def backtest_season(
    all_data: pd.DataFrame,
    test_season: str,
    adaptive: bool = True,
) -> dict:
    """
    Back-test the model on a single IPL season.

    Args:
        all_data: Complete feature dataset (all seasons)
        test_season: Season to test (e.g., "2023")
        adaptive: If True, retrain after each match result

    Returns:
        Dictionary with season back-test results
    """
    import xgboost as xgb

    # Split: train on everything before this season, test on this season
    train_data = all_data[all_data["date"] < all_data[all_data["season"] == test_season]["date"].min()]
    test_data = all_data[all_data["season"] == test_season].sort_values("date")

    if len(train_data) < 50 or len(test_data) < 5:
        return None

    feature_cols = [c for c in all_data.columns if c not in EXCLUDE_COLS and all_data[c].dtype != "object"]

    X_train = train_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_data["team1_won"].astype(int)

    # Train initial model for this season
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predict each match
    predictions = []
    actuals = []
    probabilities = []

    for idx, match in test_data.iterrows():
        X_match = pd.DataFrame([match[feature_cols]]).fillna(0).replace([np.inf, -np.inf], 0)
        y_actual = int(match["team1_won"])

        prob = model.predict_proba(X_match)[:, 1][0]
        pred = 1 if prob >= 0.5 else 0

        predictions.append(pred)
        actuals.append(y_actual)
        probabilities.append(prob)

        # Adaptive: retrain with this match's result
        if adaptive:
            new_X = X_match.values
            new_y = np.array([y_actual])
            X_train_updated = np.vstack([X_train.values, new_X])
            y_train_updated = np.concatenate([y_train.values, new_y])

            model.fit(X_train_updated, y_train_updated)
            X_train = pd.DataFrame(X_train_updated, columns=feature_cols)
            y_train = pd.Series(y_train_updated)

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    logloss = log_loss(actuals, probabilities)
    brier = brier_score_loss(actuals, probabilities)

    # Calibration: are 70% predictions actually winning ~70%?
    prob_array = np.array(probabilities)
    high_conf = prob_array >= 0.65
    if high_conf.sum() > 0:
        high_conf_acc = accuracy_score(
            np.array(actuals)[high_conf], np.array(predictions)[high_conf]
        )
    else:
        high_conf_acc = 0

    return {
        "season": test_season,
        "matches": len(test_data),
        "correct": sum(p == a for p, a in zip(predictions, actuals)),
        "accuracy": round(accuracy, 4),
        "log_loss": round(logloss, 4),
        "brier_score": round(brier, 4),
        "high_confidence_accuracy": round(high_conf_acc, 4),
        "high_confidence_matches": int(high_conf.sum()),
        "adaptive": adaptive,
    }


def run_full_backtest(seasons_to_test: int = 5):
    """
    Run back-test across multiple IPL seasons.

    Tests both static and adaptive models to show the benefit
    of in-season learning.
    """
    print("\n" + "#" * 60)
    print("#  IPL PREDICTION MODEL BACK-TESTING")
    print("#" * 60)

    features_path = PROCESSED_DATA_DIR / "match_features.csv"
    if not features_path.exists():
        print("ERROR: match_features.csv not found!")
        return

    df = pd.read_csv(features_path, parse_dates=["date"])
    df["season"] = df["season"].astype(str)

    # Get the most recent N seasons
    seasons = sorted(df["season"].unique())[-seasons_to_test:]
    print(f"\nBack-testing on seasons: {', '.join(seasons)}")

    # Run back-tests
    static_results = []
    adaptive_results = []

    for season in seasons:
        print(f"\n--- Season {season} ---")

        print("  Static model (no mid-season updates)...")
        static = backtest_season(df, season, adaptive=False)
        if static:
            static_results.append(static)
            print(f"    Accuracy: {static['accuracy']:.1%} ({static['correct']}/{static['matches']})")

        print("  Adaptive model (updates after each match)...")
        adaptive = backtest_season(df, season, adaptive=True)
        if adaptive:
            adaptive_results.append(adaptive)
            print(f"    Accuracy: {adaptive['accuracy']:.1%} ({adaptive['correct']}/{adaptive['matches']})")

    # Summary
    if static_results and adaptive_results:
        static_df = pd.DataFrame(static_results)
        adaptive_df = pd.DataFrame(adaptive_results)

        print("\n" + "=" * 60)
        print("BACK-TEST RESULTS SUMMARY")
        print("=" * 60)

        print("\n  STATIC MODEL (no mid-season updates):")
        print(f"    Overall accuracy: {static_df['accuracy'].mean():.1%}")
        print(f"    Best season:      {static_df.loc[static_df['accuracy'].idxmax(), 'season']} ({static_df['accuracy'].max():.1%})")
        print(f"    Worst season:     {static_df.loc[static_df['accuracy'].idxmin(), 'season']} ({static_df['accuracy'].min():.1%})")

        print("\n  ADAPTIVE MODEL (updates after each match):")
        print(f"    Overall accuracy: {adaptive_df['accuracy'].mean():.1%}")
        print(f"    Best season:      {adaptive_df.loc[adaptive_df['accuracy'].idxmax(), 'season']} ({adaptive_df['accuracy'].max():.1%})")
        print(f"    Worst season:     {adaptive_df.loc[adaptive_df['accuracy'].idxmin(), 'season']} ({adaptive_df['accuracy'].min():.1%})")

        improvement = adaptive_df["accuracy"].mean() - static_df["accuracy"].mean()
        print(f"\n  Adaptive vs Static improvement: {improvement:+.1%}")

        # Save results
        results_path = PROCESSED_DATA_DIR / "backtest_results.csv"
        all_results = pd.concat([static_df, adaptive_df])
        all_results.to_csv(results_path, index=False)
        print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    run_full_backtest()
