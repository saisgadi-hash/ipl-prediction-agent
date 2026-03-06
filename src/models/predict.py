"""
Make predictions for upcoming IPL matches.

This is the module you use after the model is trained.
Give it two teams and it predicts who wins.

HOW TO RUN:
    python -m src.models.predict --team1 "Chennai Super Kings" --team2 "Mumbai Indians"

    Or in Python:
        from src.models.predict import predict_match
        result = predict_match("Chennai Super Kings", "Mumbai Indians")
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "data_collection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models"))
from config import PROCESSED_DATA_DIR, MODELS_DIR
from team_name_mapper import standardise_team_name, ACTIVE_TEAMS

# The trained model pickle stores the class as '__main__.IPLEnsemblePredictor'
# because train_model.py runs as __main__. We need to make the class findable
# under __main__ when THIS script loads the pickle.
import __main__
from train_model import IPLEnsemblePredictor
__main__.IPLEnsemblePredictor = IPLEnsemblePredictor


def load_model():
    """Load the trained ensemble model and feature columns."""
    model_path = MODELS_DIR / "ensemble_predictor.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"

    if not model_path.exists():
        print("ERROR: No trained model found!")
        print("Run: python -m src.models.train_model")
        return None, None

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols


def predict_match(
    team1: str,
    team2: str,
    venue: str = None,
    toss_winner: str = None,
    toss_decision: str = None,
    skip_llm: bool = False,
) -> dict:
    """
    Predict the outcome of a match between two teams.

    Args:
        team1: Name of the first team
        team2: Name of the second team
        venue: (Optional) Venue name
        toss_winner: (Optional) Which team won the toss
        toss_decision: (Optional) "bat" or "field"

    Returns:
        Dictionary with prediction details
    """
    # Standardise team names so users can type old names like "Delhi Daredevils"
    team1 = standardise_team_name(team1)
    team2 = standardise_team_name(team2)
    if toss_winner:
        toss_winner = standardise_team_name(toss_winner)

    model, feature_cols = load_model()
    if model is None:
        return {"error": "Model not found. Train the model first."}

    # Build feature vector for this match
    # In a real scenario, this would use live data. For now, we use latest available stats.
    features = build_prediction_features(team1, team2, venue, toss_winner, toss_decision, feature_cols)

    if features is None:
        return {"error": "Could not build features for this matchup."}

    # Make prediction
    X = pd.DataFrame([features])[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    prediction = model.predict_match(X, team1, team2)

    # Add metadata
    prediction["venue"] = venue or "TBD"
    prediction["toss_winner"] = toss_winner or "TBD"
    prediction["toss_decision"] = toss_decision or "TBD"

    # Add SHAP-based justification
    try:
        from explain_prediction import explain_match_prediction
        explanation = explain_match_prediction(team1, team2, feature_vector=features)
        if "error" not in explanation:
            prediction["justification"] = explanation["text_summary"]
            prediction["top_factors"] = explanation["top_factors"]
            prediction["chart_data"] = explanation.get("chart_data")
    except Exception:
        prediction["justification"] = "Justification unavailable (SHAP not loaded)."
        prediction["top_factors"] = []
        
    # Add Phase B LLM Insights
    if not skip_llm:
        try:
            from llm_analysis import get_llm_match_analysis
            
            # Build a small context dict for the LLM
            context = {
                "venue": prediction["venue"],
                "model_predicted_winner": prediction.get('predicted_winner', 'Unknown'),
                "team1_win_prob": f"{prediction.get('team1_win_probability', 0):.1%}",
                "team2_win_prob": f"{prediction.get('team2_win_probability', 0):.1%}",
            }
            
            llm_insight = get_llm_match_analysis(team1, team2, context)
            prediction["llm_insight"] = llm_insight
        except Exception as e:
            prediction["llm_insight"] = f"LLM Insight unavailable: {str(e)}"

    return prediction


def build_prediction_features(team1, team2, venue, toss_winner, toss_decision, feature_cols):
    """
    Build a feature vector for an upcoming match using the latest available data.
    """
    # Load the most recent match features to get the latest team stats
    features_path = PROCESSED_DATA_DIR / "match_features.csv"
    if not features_path.exists():
        return None

    df = pd.read_csv(features_path)

    # Find the most recent match for each team to get their latest features
    team1_recent = df[(df["team1"] == team1) | (df["team2"] == team1)].iloc[-1:] if len(df[(df["team1"] == team1) | (df["team2"] == team1)]) > 0 else None
    team2_recent = df[(df["team1"] == team2) | (df["team2"] == team2)].iloc[-1:] if len(df[(df["team1"] == team2) | (df["team2"] == team2)]) > 0 else None

    # Start with zeros for all features
    features = {col: 0 for col in feature_cols}

    # Fill in what we can from recent data
    if team1_recent is not None and len(team1_recent) > 0:
        row = team1_recent.iloc[0]
        for col in feature_cols:
            if col in row.index and pd.notna(row[col]):
                try:
                    features[col] = float(row[col])
                except (ValueError, TypeError):
                    features[col] = 0

    # Override toss features if provided
    if toss_winner:
        features["toss_winner_is_team1"] = 1 if toss_winner == team1 else 0
    if toss_decision:
        features["toss_decision_bat"] = 1 if toss_decision == "bat" else 0

    return features


def predict_tournament_winner(teams: list = None) -> list:
    """
    Predict the most likely IPL tournament winner.

    Simulates all possible matchups and ranks teams by overall win probability.
    """
    model, feature_cols = load_model()
    if model is None:
        return []

    if teams is None:
        teams = ACTIVE_TEAMS.copy()

    # Calculate average win probability against all other teams
    team_scores = {}
    for team in teams:
        total_prob = 0
        matches = 0
        for opponent in teams:
            if opponent == team:
                continue
            # Skip LLM for tournament predictions to avoid API rate limits
            result = predict_match(team, opponent, skip_llm=True)
            if "error" not in result:
                total_prob += result["team1_win_probability"]
                matches += 1

        avg_prob = total_prob / max(matches, 1)
        team_scores[team] = round(avg_prob, 4)

    # Sort by probability (descending)
    rankings = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"rank": i + 1, "team": team, "win_probability": prob}
        for i, (team, prob) in enumerate(rankings)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict IPL match outcome")
    parser.add_argument("--team1", type=str, default="Chennai Super Kings")
    parser.add_argument("--team2", type=str, default="Mumbai Indians")
    parser.add_argument("--venue", type=str, default=None)
    parser.add_argument("--tournament", action="store_true", help="Predict tournament winner")

    args = parser.parse_args()

    if args.tournament:
        print("\n" + "=" * 50)
        print("IPL TOURNAMENT WINNER PREDICTION")
        print("=" * 50)
        rankings = predict_tournament_winner()
        for r in rankings:
            bar = "#" * int(r["win_probability"] * 50)
            print(f"  {r['rank']:2d}. {r['team']:35s} {r['win_probability']:.1%} {bar}")
    else:
        print(f"\nPredicting: {args.team1} vs {args.team2}")
        result = predict_match(args.team1, args.team2, args.venue)
        print(f"\n  Predicted Winner: {result.get('predicted_winner', 'N/A')}")
        print(f"  {result.get('team1', 'Team 1')} Win Probability: {result.get('team1_win_probability', 0):.1%}")
        print(f"  {result.get('team2', 'Team 2')} Win Probability: {result.get('team2_win_probability', 0):.1%}")
        print(f"  Confidence: {result.get('confidence', 0):.0f}%")

        # Show justification
        if result.get("justification"):
            print(f"\n{'='*50}")
            print("STATISTICAL JUSTIFICATION (SHAP)")
            print(f"{'-'*50}")
            print(result["justification"])
            
        # Show LLM Insight
        if result.get("llm_insight"):
            print(f"\n{'='*50}")
            print("AI TACTICAL PREVIEW (LLM)")
            print(f"{'-'*50}")
            print(result["llm_insight"])
            print(f"{'='*50}")
