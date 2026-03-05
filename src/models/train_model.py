"""
Train the IPL match prediction model.

This is where the machine learning magic happens. We train an ensemble
of models (XGBoost, LightGBM, Logistic Regression) on the feature-engineered
data to predict match outcomes.

HOW TO RUN:
    python -m src.models.train_model

WHAT IT DOES:
    1. Loads the feature-engineered dataset (match_features.csv)
    2. Splits data chronologically (train on older data, test on newer)
    3. Trains XGBoost, LightGBM, and Logistic Regression models
    4. Combines them into an ensemble
    5. Evaluates accuracy and generates SHAP explanations
    6. Saves the trained model to models/

BEGINNER NOTES:
    - "Training" means the model studies historical data to find patterns
    - "Testing" means we check if the model can predict matches it hasn't seen
    - We always split chronologically: train on past, test on future
    - "Ensemble" means combining multiple models for better accuracy
    - SHAP values explain WHY the model made each prediction
"""

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import PROCESSED_DATA_DIR, MODELS_DIR, MODEL_CONFIG


# ══════════════════════════════════════════
# STEP 1: DATA PREPARATION
# ══════════════════════════════════════════

# Columns to EXCLUDE from model features (metadata, targets, IDs)
EXCLUDE_COLS = [
    "match_id", "date", "team1", "team2", "winner", "team1_won",
    "season", "venue", "city",
]


def load_and_prepare_data():
    """
    Load feature data and split into train/test sets.

    CRITICAL: We split CHRONOLOGICALLY, not randomly.
    Train on older matches, test on newer ones.
    This mimics real-world usage where you predict future matches.
    """
    features_path = PROCESSED_DATA_DIR / "match_features.csv"

    if not features_path.exists():
        print("ERROR: match_features.csv not found!")
        print("Run: python -m src.features.build_match_features")
        return None, None, None, None, None

    print("Loading feature data...")
    df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"  Total matches: {len(df)}")
    print(f"  Total features: {len(df.columns)}")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Target variable
    y = df["team1_won"].astype(int)

    # Feature columns (everything except excluded)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Handle any remaining non-numeric columns
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X = X.drop(columns=[col])
            feature_cols.remove(col)

    # Fill NaN with 0 (safe default for missing features)
    X = X.fillna(0)

    # Replace infinity values
    X = X.replace([np.inf, -np.inf], 0)

    # Chronological split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\n  Training set: {len(X_train)} matches (older)")
    print(f"  Test set:     {len(X_test)} matches (newer)")
    print(f"  Features:     {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, feature_cols


# ══════════════════════════════════════════
# STEP 2: TRAIN INDIVIDUAL MODELS
# ══════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier.

    XGBoost (Extreme Gradient Boosting) builds many small decision trees,
    each one learning from the mistakes of the previous ones.
    It's the most popular algorithm for tabular data.
    """
    import xgboost as xgb

    print("\n--- Training XGBoost ---")

    xgb_config = MODEL_CONFIG.get("xgboost", {})

    model = xgb.XGBClassifier(
        n_estimators=xgb_config.get("n_estimators", 500),
        max_depth=xgb_config.get("max_depth", 6),
        learning_rate=xgb_config.get("learning_rate", 0.05),
        subsample=xgb_config.get("subsample", 0.8),
        colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
        min_child_weight=xgb_config.get("min_child_weight", 3),
        reg_alpha=xgb_config.get("reg_alpha", 0.1),
        reg_lambda=xgb_config.get("reg_lambda", 1.0),
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=50,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)

    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Log Loss: {log_loss(y_test, probabilities):.4f}")

    return model, accuracy


def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    Train a LightGBM classifier.

    LightGBM is similar to XGBoost but faster and often just as accurate.
    Good to have as a second opinion in the ensemble.
    """
    import lightgbm as lgb

    print("\n--- Training LightGBM ---")

    lgb_config = MODEL_CONFIG.get("lightgbm", {})

    model = lgb.LGBMClassifier(
        n_estimators=lgb_config.get("n_estimators", 500),
        max_depth=lgb_config.get("max_depth", 6),
        learning_rate=lgb_config.get("learning_rate", 0.05),
        subsample=lgb_config.get("subsample", 0.8),
        colsample_bytree=lgb_config.get("colsample_bytree", 0.8),
        num_leaves=lgb_config.get("num_leaves", 31),
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)

    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Log Loss: {log_loss(y_test, probabilities):.4f}")

    return model, accuracy


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train a Logistic Regression classifier.

    Logistic Regression is the simplest model. It's our baseline.
    If the ensemble can't beat this, something is wrong with our features.
    """
    print("\n--- Training Logistic Regression (Baseline) ---")

    # Scale features (LR needs normalised inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, predictions)

    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Log Loss: {log_loss(y_test, probabilities):.4f}")

    return model, scaler, accuracy


# ══════════════════════════════════════════
# STEP 3: BUILD ENSEMBLE
# ══════════════════════════════════════════

class IPLEnsemblePredictor:
    """
    Ensemble model that combines XGBoost, LightGBM, and Logistic Regression.

    The ensemble takes the weighted average of all three models' predictions.
    This usually performs better than any single model because different
    models have different strengths and weaknesses.

    BEGINNER NOTE:
        Think of it like asking three cricket experts for their opinion
        and going with the weighted average. The expert who's been most
        accurate gets more weight.
    """

    def __init__(self, xgb_model, lgb_model, lr_model, lr_scaler, weights=None):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.lr_model = lr_model
        self.lr_scaler = lr_scaler

        ensemble_config = MODEL_CONFIG.get("ensemble", {})
        self.weights = weights or [
            ensemble_config.get("xgboost_weight", 0.45),
            ensemble_config.get("lightgbm_weight", 0.35),
            ensemble_config.get("logistic_weight", 0.20),
        ]

    def predict_proba(self, X):
        """Get win probability from the ensemble."""
        X_scaled = self.lr_scaler.transform(X)

        prob_xgb = self.xgb_model.predict_proba(X)[:, 1]
        prob_lgb = self.lgb_model.predict_proba(X)[:, 1]
        prob_lr = self.lr_model.predict_proba(X_scaled)[:, 1]

        ensemble_prob = (
            prob_xgb * self.weights[0] +
            prob_lgb * self.weights[1] +
            prob_lr * self.weights[2]
        )
        return ensemble_prob

    def predict(self, X):
        """Get binary prediction (0 or 1) from the ensemble."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_match(self, X, team1_name, team2_name):
        """
        Make a human-readable prediction for a match.

        Returns a dictionary with the prediction and explanation.
        """
        proba = self.predict_proba(X)[0]

        return {
            "team1": team1_name,
            "team2": team2_name,
            "team1_win_probability": round(float(proba), 4),
            "team2_win_probability": round(1 - float(proba), 4),
            "predicted_winner": team1_name if proba >= 0.5 else team2_name,
            "confidence": round(abs(proba - 0.5) * 200, 1),  # 0-100% confidence
        }


# ══════════════════════════════════════════
# STEP 4: SHAP EXPLAINABILITY
# ══════════════════════════════════════════

def explain_model(model, X_test, feature_names, n_explanations=5):
    """
    Generate SHAP explanations for model predictions.

    SHAP (SHapley Additive exPlanations) tells you exactly WHY
    the model predicted what it did for each match.
    """
    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        return None

    print("\n--- Generating SHAP Explanations ---")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.head(n_explanations))

    print(f"\n  Top features driving predictions (first {n_explanations} matches):")

    for i in range(min(n_explanations, len(shap_values))):
        print(f"\n  Match {i + 1}:")
        vals = shap_values[i]
        top_features = sorted(
            zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        for name, val in top_features:
            direction = "favours Team 1" if val > 0 else "favours Team 2"
            print(f"    {name}: {val:+.4f} ({direction})")

    return shap_values


# ══════════════════════════════════════════
# STEP 5: MAIN TRAINING PIPELINE
# ══════════════════════════════════════════

def train_full_pipeline():
    """
    Run the complete model training pipeline.

    This is the main function. Run it and it does everything:
    load data → train models → build ensemble → evaluate → explain → save.
    """
    print("\n" + "#" * 60)
    print("#  IPL PREDICTION MODEL TRAINING")
    print("#" * 60)

    # Load data
    result = load_and_prepare_data()
    if result[0] is None:
        return

    X_train, X_test, y_train, y_test, feature_cols = result

    # Train individual models
    xgb_model, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_model, lgb_acc = train_lightgbm(X_train, y_train, X_test, y_test)
    lr_model, lr_scaler, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)

    # Build ensemble
    print("\n--- Building Ensemble ---")
    ensemble = IPLEnsemblePredictor(xgb_model, lgb_model, lr_model, lr_scaler)

    ensemble_preds = ensemble.predict(X_test)
    ensemble_proba = ensemble.predict_proba(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)

    print(f"  Ensemble Accuracy: {ensemble_acc:.1%}")
    print(f"  Ensemble Log Loss: {log_loss(y_test, ensemble_proba):.4f}")
    print(f"  Ensemble Brier Score: {brier_score_loss(y_test, ensemble_proba):.4f}")

    # Detailed classification report
    print("\n--- Detailed Results ---")
    print(classification_report(y_test, ensemble_preds, target_names=["Team 2 Wins", "Team 1 Wins"]))

    # SHAP explanations (using XGBoost component)
    explain_model(xgb_model, X_test, feature_cols)

    # ── Save everything ──
    print("\n--- Saving Models ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "ensemble_predictor.pkl"
    joblib.dump(ensemble, model_path)
    print(f"  Ensemble saved to: {model_path}")

    # Save feature columns (needed for prediction)
    feature_path = MODELS_DIR / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_path)
    print(f"  Feature columns saved to: {feature_path}")

    # Save individual models too
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.pkl")
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm_model.pkl")
    joblib.dump(lr_model, MODELS_DIR / "logistic_model.pkl")
    joblib.dump(lr_scaler, MODELS_DIR / "lr_scaler.pkl")

    # Save training metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
        "xgboost_accuracy": round(xgb_acc, 4),
        "lightgbm_accuracy": round(lgb_acc, 4),
        "logistic_accuracy": round(lr_acc, 4),
        "ensemble_accuracy": round(ensemble_acc, 4),
    }
    joblib.dump(metadata, MODELS_DIR / "training_metadata.pkl")

    # Summary
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  XGBoost Accuracy:   {xgb_acc:.1%}")
    print(f"  LightGBM Accuracy:  {lgb_acc:.1%}")
    print(f"  Logistic Accuracy:  {lr_acc:.1%}")
    print(f"  ENSEMBLE Accuracy:  {ensemble_acc:.1%}")

    target_acc = MODEL_CONFIG.get("accuracy", {}).get("target", 0.65)
    if ensemble_acc >= target_acc:
        print(f"\n  TARGET MET! ({target_acc:.0%})")
    else:
        print(f"\n  Below target ({target_acc:.0%}). Consider adding more features.")

    print(f"\nAll models saved to: {MODELS_DIR}/")
    print(f"\nNext step: Run  python -m src.models.backtest")


if __name__ == "__main__":
    train_full_pipeline()
