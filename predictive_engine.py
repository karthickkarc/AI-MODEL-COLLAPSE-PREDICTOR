"""
predictive_engine.py
---------------------
Trains and serves the two core predictive models:

1. Risk Classifier (RandomForest) - predicts current Risk Level
   (Low/Medium/High) from the live feature vector. This is learned from
   data rather than hard-coded thresholds, so it can be retrained on real
   labeled outcomes as they accumulate.

2. Forecasting Regressors (XGBoost) - one per horizon (7-day, 30-day) -
   predict the probability that the model will have collapsed
   (accuracy < 0.5) within that horizon, given today's feature vector.

Run directly to train from data/simulated_model_generations.csv and save
artifacts to models/.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__)))
from feature_engineering import FEATURE_COLUMNS  # noqa: E402

THIS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(THIS_DIR, "..", "models")
DATA_PATH = os.path.join(THIS_DIR, "..", "data", "simulated_model_generations.csv")

HORIZONS = (7, 30)


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df


def train_risk_classifier(df):
    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("\n=== Risk Classifier (RandomForest) ===")
    print(classification_report(y_test, preds))
    return clf


def train_forecast_models(df):
    """One XGBoost binary classifier per horizon predicting collapse_event_Hd."""
    forecasters = {}
    for h in HORIZONS:
        label_col = f"collapse_event_{h}d"
        sub = df.dropna(subset=[label_col])
        X = sub[FEATURE_COLUMNS]
        y = sub[label_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = float("nan")
        print(f"\n=== Forecast model ({h}-day horizon) === AUC: {auc:.3f}")
        forecasters[h] = model
    return forecasters


def collapse_probability_today(features_row):
    """
    'Today' risk isn't a forecast -- it's derived directly from the current
    health score via a logistic mapping, so the dashboard always has a
    Day-0 number even though the ML forecasters only cover 7/30 day horizons.
    """
    from feature_engineering import health_score
    h = health_score(
        features_row["diversity"], features_row["entropy_norm"],
        features_row["drift_norm"], features_row["accuracy"],
        features_row["synthetic_ratio"],
    )
    # Map health score (0-100, higher=healthier) to a 0-1 collapse probability
    return float(np.clip((100 - h) / 100, 0, 1))


def predict_risk(clf, features_row: dict):
    X = pd.DataFrame([features_row])[FEATURE_COLUMNS]
    pred = clf.predict(X)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(X)[0]))
    return pred, proba


def predict_forecast(forecasters, features_row: dict):
    X = pd.DataFrame([features_row])[FEATURE_COLUMNS]
    out = {0: round(collapse_probability_today(features_row) * 100, 1)}
    for h, model in forecasters.items():
        p = model.predict_proba(X)[0, 1]
        out[h] = round(float(p) * 100, 1)
    return out  # {0: today%, 7: +7day%, 30: +30day%}


def save_artifacts(clf, forecasters):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "risk_classifier.joblib"))
    for h, model in forecasters.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"forecast_{h}d.joblib"))
    print(f"\nSaved models to {os.path.abspath(MODELS_DIR)}")


def load_artifacts():
    clf = joblib.load(os.path.join(MODELS_DIR, "risk_classifier.joblib"))
    forecasters = {
        h: joblib.load(os.path.join(MODELS_DIR, f"forecast_{h}d.joblib"))
        for h in HORIZONS
    }
    return clf, forecasters


def main():
    df = load_dataset()
    clf = train_risk_classifier(df)
    forecasters = train_forecast_models(df)
    save_artifacts(clf, forecasters)


if __name__ == "__main__":
    main()
