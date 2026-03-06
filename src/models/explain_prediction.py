"""
Prediction Justification — Explains WHY the model predicts a certain winner.

Uses SHAP (SHapley Additive exPlanations) to break down each prediction
into human-readable reasons, like:

    "CSK are predicted to win because:
     1. CSK's recent form is much stronger (won 4 of last 5)
     2. CSK have a strong head-to-head record at this venue (72% win rate)
     3. MI's top bowlers are in poor form (economy 9.5+ in last 3 matches)
     4. Toss advantage: CSK won toss and chose to field (65% chase-win rate here)"

HOW TO USE:
    from src.models.explain_prediction import explain_match_prediction

    explanation = explain_match_prediction("CSK", "MI")
    print(explanation["text_summary"])
    # Also returns: shap_values, top_factors, chart_data

BEGINNER NOTES:
    - SHAP values are like "credit scores" for each feature
    - Positive SHAP = helps Team 1, Negative SHAP = helps Team 2
    - The bigger the absolute value, the more important that feature is
    - We translate these numbers into plain English explanations
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "data_collection"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models"))
from config import MODELS_DIR, PROCESSED_DATA_DIR
from team_name_mapper import standardise_team_name, get_short_code
import __main__
from train_model import IPLEnsemblePredictor
__main__.IPLEnsemblePredictor = IPLEnsemblePredictor


# ══════════════════════════════════════════
# FEATURE NAME → HUMAN-READABLE TRANSLATIONS
# ══════════════════════════════════════════

FEATURE_TRANSLATIONS = {
    # Recent form
    "team1_recent_win_pct": "{team1} recent win rate",
    "team2_recent_win_pct": "{team2} recent win rate",
    "form_win_pct_diff": "Recent form difference",
    "team1_recent_wins": "{team1} recent wins",
    "team2_recent_wins": "{team2} recent wins",
    "team1_recent_avg_margin": "{team1} average winning margin",
    "team2_recent_avg_margin": "{team2} average winning margin",

    # Season standings
    "team1_season_win_pct": "{team1} season win percentage",
    "team2_season_win_pct": "{team2} season win percentage",
    "points_diff": "Points table difference",
    "team1_season_points": "{team1} points in season",
    "team2_season_points": "{team2} points in season",

    # Head to head
    "h2h_team1_win_pct": "Head-to-head win rate",
    "h2h_total_matches": "Head-to-head matches played",
    "h2h_team1_wins": "{team1} H2H wins",
    "h2h_team2_wins": "{team2} H2H wins",

    # Venue
    "team1_venue_win_pct": "{team1} win rate at this venue",
    "team2_venue_win_pct": "{team2} win rate at this venue",
    "venue_win_pct_diff": "Venue advantage difference",
    "venue_avg_1st_innings": "Average 1st innings score at venue",
    "venue_avg_2nd_innings": "Average 2nd innings score at venue",
    "venue_chase_win_pct": "Chase success rate at venue",

    # Toss
    "toss_winner_is_team1": "Toss result",
    "toss_decision_bat": "Toss decision (bat/field)",

    # Player form
    "team1_top_batter_form_avg": "{team1} key batters' form",
    "team2_top_batter_form_avg": "{team2} key batters' form",
    "team1_top_bowler_form_avg": "{team1} key bowlers' form",
    "team2_top_bowler_form_avg": "{team2} key bowlers' form",

    # Match context
    "match_number_in_season": "Stage of the tournament",
}


def get_feature_label(feature_name: str, team1: str, team2: str) -> str:
    """Convert a feature column name to a human-readable label."""
    code1 = get_short_code(team1)
    code2 = get_short_code(team2)

    if feature_name in FEATURE_TRANSLATIONS:
        return FEATURE_TRANSLATIONS[feature_name].format(team1=code1, team2=code2)

    # Fallback: clean up the column name
    label = feature_name.replace("_", " ").replace("team1", code1).replace("team2", code2)
    return label.title()


def generate_text_reason(feature_name: str, shap_value: float, feature_value: float,
                         team1: str, team2: str) -> str:
    """
    Generate a plain English sentence explaining one factor.

    This is the heart of the justification system — it turns numbers
    into sentences that anyone can understand.
    """
    code1 = get_short_code(team1)
    code2 = get_short_code(team2)
    favours = code1 if shap_value > 0 else code2
    impact = abs(shap_value)

    # Strength of the factor
    if impact > 0.15:
        strength = "strongly"
    elif impact > 0.08:
        strength = "moderately"
    else:
        strength = "slightly"

    # ── Form-related ──
    if "recent_win_pct" in feature_name and "diff" in feature_name:
        if shap_value > 0:
            return f"{code1}'s recent form is {strength} better than {code2}'s"
        else:
            return f"{code2}'s recent form is {strength} better than {code1}'s"

    if "recent_win_pct" in feature_name:
        team = code1 if "team1" in feature_name else code2
        pct = round(feature_value * 100)
        return f"{team} have won {pct}% of their recent matches"

    # ── Head-to-head ──
    if "h2h_team1_win_pct" in feature_name:
        if shap_value > 0:
            return f"{code1} have a {strength} superior head-to-head record against {code2}"
        else:
            return f"{code2} have a {strength} superior head-to-head record against {code1}"

    # ── Venue ──
    if "venue_win_pct" in feature_name and "diff" in feature_name:
        return f"Venue history {strength} favours {favours}"

    if "venue_win_pct" in feature_name:
        team = code1 if "team1" in feature_name else code2
        pct = round(feature_value * 100)
        return f"{team} have a {pct}% win rate at this venue"

    if "venue_chase_win_pct" in feature_name:
        pct = round(feature_value * 100)
        return f"Teams chasing win {pct}% of the time at this venue"

    # ── Points table ──
    if "points_diff" in feature_name:
        if shap_value > 0:
            return f"{code1} are higher on the points table"
        else:
            return f"{code2} are higher on the points table"

    if "season_win_pct" in feature_name:
        team = code1 if "team1" in feature_name else code2
        pct = round(feature_value * 100)
        return f"{team}'s season win rate is {pct}%"

    # ── Toss ──
    if "toss_winner_is_team1" in feature_name:
        winner = code1 if feature_value == 1 else code2
        return f"{winner} won the toss, which {strength} helps them"

    if "toss_decision_bat" in feature_name:
        decision = "batted first" if feature_value == 1 else "chose to field"
        return f"Toss winner {decision}"

    # ── Player form ──
    if "batter_form" in feature_name:
        team = code1 if "team1" in feature_name else code2
        if ("team1" in feature_name and shap_value > 0) or ("team2" in feature_name and shap_value < 0):
            return f"{team}'s key batters are in {strength.replace('ly', '')} form"
        else:
            return f"{team}'s key batters are {strength} out of form"

    if "bowler_form" in feature_name:
        team = code1 if "team1" in feature_name else code2
        if ("team1" in feature_name and shap_value > 0) or ("team2" in feature_name and shap_value < 0):
            return f"{team}'s key bowlers are in {strength.replace('ly', '')} form"
        else:
            return f"{team}'s key bowlers are {strength} struggling"

    # ── Fallback ──
    label = get_feature_label(feature_name, team1, team2)
    return f"{label} {strength} favours {favours}"


def explain_match_prediction(team1: str, team2: str, feature_vector: dict = None) -> dict:
    """
    Generate a full prediction explanation for a match.

    Args:
        team1: First team name (any historical variant accepted)
        team2: Second team name
        feature_vector: Pre-computed features (optional; if None, uses latest data)

    Returns:
        Dictionary with:
            - predicted_winner: str
            - win_probability: float
            - confidence: float (0-100)
            - top_factors: list of {feature, shap_value, label, reason, direction}
            - text_summary: str (multi-line human-readable explanation)
            - chart_data: dict (for Plotly waterfall chart)
    """
    team1 = standardise_team_name(team1)
    team2 = standardise_team_name(team2)
    code1 = get_short_code(team1)
    code2 = get_short_code(team2)

    # Load models
    model_path = MODELS_DIR / "ensemble_predictor.pkl"
    xgb_path = MODELS_DIR / "xgboost_model.pkl"
    features_path = MODELS_DIR / "feature_columns.pkl"

    if not model_path.exists():
        return {"error": "No trained model found. Run: python -m src.models.train_model"}

    ensemble = joblib.load(model_path)
    xgb_model = joblib.load(xgb_path)
    feature_cols = joblib.load(features_path)

    # Build features if not provided
    if feature_vector is None:
        from predict import build_prediction_features
        feature_vector = build_prediction_features(team1, team2, None, None, None, feature_cols)
        if feature_vector is None:
            return {"error": "Could not build features for this matchup."}

    # Create DataFrame for prediction
    X = pd.DataFrame([feature_vector])[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Get ensemble prediction
    prediction = ensemble.predict_match(X, team1, team2)
    win_prob = prediction["team1_win_probability"]
    winner = prediction["predicted_winner"]
    confidence = prediction["confidence"]

    # Get SHAP values from XGBoost (tree-based SHAP is most reliable)
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Class 1 (team1 wins)
        else:
            shap_vals = shap_values[0]
    except Exception:
        # Fallback: use feature importances if SHAP fails
        importances = xgb_model.feature_importances_
        shap_vals = importances * np.sign(X.values[0] - X.values[0].mean())

    # Build top factors
    feature_shap_pairs = list(zip(feature_cols, shap_vals, X.values[0]))
    feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = []
    for feat_name, shap_val, feat_val in feature_shap_pairs[:10]:
        direction = "favours_team1" if shap_val > 0 else "favours_team2"
        reason = generate_text_reason(feat_name, shap_val, feat_val, team1, team2)
        label = get_feature_label(feat_name, team1, team2)

        top_factors.append({
            "feature": feat_name,
            "shap_value": round(float(shap_val), 4),
            "feature_value": round(float(feat_val), 4),
            "label": label,
            "reason": reason,
            "direction": direction,
        })

    # Build text summary
    winner_code = get_short_code(winner)
    loser = team2 if winner == team1 else team1
    loser_code = get_short_code(loser)
    win_pct = win_prob if winner == team1 else (1 - win_prob)

    lines = [
        f"PREDICTION: {winner_code} to beat {loser_code}",
        f"Win Probability: {win_pct:.0%} | Confidence: {confidence:.0f}%",
        "",
        "KEY REASONS:",
    ]

    for i, factor in enumerate(top_factors[:5], 1):
        emoji = "+" if factor["direction"] == ("favours_team1" if winner == team1 else "favours_team2") else "-"
        lines.append(f"  {i}. [{emoji}] {factor['reason']}")

    text_summary = "\n".join(lines)

    # Build chart data (for Plotly waterfall chart in dashboard)
    chart_data = {
        "labels": [f["label"] for f in top_factors[:8]],
        "values": [f["shap_value"] for f in top_factors[:8]],
        "colors": [
            "#00C853" if f["shap_value"] > 0 else "#FF1744"
            for f in top_factors[:8]
        ],
        "base_value": float(explainer.expected_value) if 'explainer' in dir() else 0.5,
    }

    return {
        "predicted_winner": winner,
        "win_probability": round(win_pct, 4),
        "confidence": round(confidence, 1),
        "team1": team1,
        "team2": team2,
        "top_factors": top_factors,
        "text_summary": text_summary,
        "chart_data": chart_data,
        "shap_values": {feat: round(float(val), 4) for feat, val in zip(feature_cols, shap_vals)},
    }


def print_explanation(team1: str, team2: str):
    """Print a formatted prediction explanation to the console."""
    result = explain_match_prediction(team1, team2)

    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "=" * 60)
    print(result["text_summary"])
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explain an IPL match prediction")
    parser.add_argument("--team1", type=str, default="Chennai Super Kings")
    parser.add_argument("--team2", type=str, default="Mumbai Indians")
    args = parser.parse_args()

    print_explanation(args.team1, args.team2)
