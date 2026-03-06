import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Attempting to safely import config constants
try:
    from config import PROCESSED_DATA_DIR, MODELS_DIR
except ImportError:
    # Fallback to local paths if config is missing
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    MODELS_DIR = os.path.join(BASE_DIR, "data", "models")


def train_model():
    print("=" * 60)
    print(" TRAIN MODEL ")
    print("=" * 60)
    
    data_path = os.path.join(PROCESSED_DATA_DIR, "match_features.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please run `python -m src.features.build_match_features` first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} matches.")
    
    # Exclude non-numeric or purely identifer strings from basic features
    exclude_cols = ["match_id", "date", "team1", "team2", "winner", "team1_won", "venue", "city"]
    
    features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'bool', 'int', 'float']]
    target = "team1_won"
    
    # Drop columns that are completely empty
    df = df.dropna(axis=1, how='all')
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = df[target]
    
    # Fill missing values
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # Train-Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} instances...")
    
    # Use Random Forest Classifier for an initial baseline
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    
    # Feature Importance display
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save Model Weights and metadata
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    joblib.dump(model, model_path)
    
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.joblib")
    joblib.dump({"features": features, "imputer": imputer}, metadata_path)
    
    print(f"\nModel and features metadata saved to {MODELS_DIR}")


if __name__ == "__main__":
    train_model()
