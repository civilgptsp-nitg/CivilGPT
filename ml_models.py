# ml_models.py — CivilGPT v1.7 AI Models (Strength + Slump Predictors)

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===============================
# Paths for saved models
# ===============================
STRENGTH_MODEL_PATH = "models/strength_model.pkl"
SLUMP_MODEL_PATH = "models/slump_model.pkl"

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# ===============================
# Training Functions
# ===============================
def train_strength_model(data_path: str):
    """
    Train compressive strength model.
    Expects dataset with columns: Cement, Water, FineAgg, CoarseAgg, SCMs, Age, Strength
    """
    df = pd.read_excel(data_path)

    # Basic sanity check
    if "Strength" not in df.columns:
        raise ValueError("Dataset must contain 'Strength' column (MPa).")

    X = df.drop(columns=["Strength"])
    y = df["Strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Strength Model R²:", r2_score(y_test, preds))
    print("Strength Model MAE:", mean_absolute_error(y_test, preds))

    joblib.dump(model, STRENGTH_MODEL_PATH)
    print(f"✅ Strength model saved at {STRENGTH_MODEL_PATH}")


def train_slump_model(data_path: str):
    """
    Train slump model.
    Expects dataset with columns: Cement, Water, FineAgg, CoarseAgg, SCMs, SP, Slump
    """
    df = pd.read_excel(data_path)

    if "Slump" not in df.columns:
        raise ValueError("Dataset must contain 'Slump' column (mm).")

    X = df.drop(columns=["Slump"])
    y = df["Slump"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Slump Model R²:", r2_score(y_test, preds))
    print("Slump Model MAE:", mean_absolute_error(y_test, preds))

    joblib.dump(model, SLUMP_MODEL_PATH)
    print(f"✅ Slump model saved at {SLUMP_MODEL_PATH}")

# ===============================
# Load + Predict Functions
# ===============================
def load_models():
    strength_model = joblib.load(STRENGTH_MODEL_PATH) if os.path.exists(STRENGTH_MODEL_PATH) else None
    slump_model = joblib.load(SLUMP_MODEL_PATH) if os.path.exists(SLUMP_MODEL_PATH) else None
    return strength_model, slump_model

def predict_strength(mix_features: dict, model=None):
    if model is None:
        model, _ = load_models()
    if model is None:
        raise RuntimeError("Strength model not trained yet.")
    df = pd.DataFrame([mix_features])
    return float(model.predict(df)[0])

def predict_slump(mix_features: dict, model=None):
    _, model_slump = load_models()
    if model is None:
        model = model_slump
    if model is None:
        raise RuntimeError("Slump model not trained yet.")
    df = pd.DataFrame([mix_features])
    return float(model.predict(df)[0])
