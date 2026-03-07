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
    """Load the trained ensemble model, feature columns, and optional calibrator."""
    model_path = MODELS_DIR / "ensemble_predictor.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"
    calibrator_path = MODELS_DIR / "probability_calibrator.pkl"

    if not model_path.exists():
        print("ERROR: No trained model found!")
        print("Run: python -m src.models.train_model")
        return None, None, None

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    calibrator = None
    if calibrator_path.exists():
        try:
            calibrator = joblib.load(calibrator_path)
        except Exception:
            pass

    return model, feature_cols, calibrator


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

    model, feature_cols, calibrator = load_model()
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

    # Apply probability calibration if available (Phase B)
    if calibrator is not None:
        try:
            raw_prob = prediction["team1_win_probability"]
            calibrated = float(calibrator.predict([raw_prob])[0])
            prediction["team1_win_probability"] = round(calibrated, 4)
            prediction["team2_win_probability"] = round(1 - calibrated, 4)
            prediction["predicted_winner"] = team1 if calibrated >= 0.5 else team2
            prediction["confidence"] = round(abs(calibrated - 0.5) * 200, 1)
            prediction["calibrated"] = True
        except Exception:
            prediction["calibrated"] = False
    else:
        prediction["calibrated"] = False

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


def _get_current_elo_ratings(teams: list) -> dict:
    """
    Extract current Elo ratings for all teams from historical match data.
    Uses the latest date as the 'current' reference point.
    """
    matches_path = PROCESSED_DATA_DIR / "matches.csv"
    if not matches_path.exists():
        # Fallback: return default ratings
        return {team: 1500.0 for team in teams}

    matches = pd.read_csv(matches_path)
    matches['date'] = pd.to_datetime(matches['date'])

    # Use a date far in the future so all matches are included
    future_date = pd.Timestamp('2030-01-01')

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "features"))
    from elo_ratings import calculate_elo

    elo_ratings = {}
    for team in teams:
        elo_ratings[team] = calculate_elo(matches, team, future_date)

    return elo_ratings


def _get_current_hmm_states(teams: list) -> dict:
    """
    Get current HMM form state (0=Cold, 1=Normal, 2=Hot) for each team.
    """
    matches_path = PROCESSED_DATA_DIR / "matches.csv"
    if not matches_path.exists():
        return {team: 1 for team in teams}

    matches = pd.read_csv(matches_path)
    matches['date'] = pd.to_datetime(matches['date'])
    future_date = pd.Timestamp('2030-01-01')

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "features"))
    from hmm_form import calculate_hmm_state

    states = {}
    for team in teams:
        try:
            states[team] = calculate_hmm_state(matches, team, future_date)
        except Exception:
            states[team] = 1  # Default to Normal
    return states


# Cache for Monte Carlo results (avoid recalculating on every request)
_tournament_cache = {"result": None, "timestamp": 0}
_CACHE_TTL = 300  # 5 minutes

import time


def predict_tournament_winner(teams: list = None, num_simulations: int = 5000) -> list:
    """
    Predict tournament winner using Monte Carlo simulation with Elo-based probabilities.

    Runs N full season simulations (double round-robin + playoffs) and returns
    win probability distributions with confidence intervals.
    """
    global _tournament_cache

    # Check cache
    now = time.time()
    if _tournament_cache["result"] is not None and (now - _tournament_cache["timestamp"]) < _CACHE_TTL:
        return _tournament_cache["result"]

    if teams is None:
        teams = list(ACTIVE_TEAMS.copy())

    # Get current Elo ratings from historical data
    elo_ratings = _get_current_elo_ratings(teams)

    # Get HMM form states
    hmm_states = _get_current_hmm_states(teams)
    state_labels = {0: "Cold", 1: "Normal", 2: "Hot"}

    # Run Monte Carlo simulation
    from tournament_simulation import simulate_tournament
    probabilities = simulate_tournament(teams, elo_ratings, num_simulations)

    # Build ranked results with advanced stats
    sorted_teams = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    rankings = []
    for i, (team, prob) in enumerate(sorted_teams):
        rankings.append({
            "rank": i + 1,
            "team": team,
            "win_probability": round(prob, 4),
            "elo_rating": round(elo_ratings.get(team, 1500), 1),
            "form_state": state_labels.get(hmm_states.get(team, 1), "Normal"),
            "simulations": num_simulations,
        })

    # Cache the result
    _tournament_cache = {"result": rankings, "timestamp": now}

    return rankings


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
