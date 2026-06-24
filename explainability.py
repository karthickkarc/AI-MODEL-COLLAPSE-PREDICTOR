"""
explainability.py
------------------
Wraps a trained RandomForest risk classifier with a SHAP TreeExplainer so
every prediction can be accompanied by a human-readable cause, e.g.:

    "Risk increased mainly because: Diversity dropped, Synthetic data ratio
     rose, Drift increased."

Run this file directly for a quick demo against the trained models.
"""

import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.dirname(__file__))
from feature_engineering import FEATURE_COLUMNS  # noqa: E402

FRIENDLY_NAMES = {
    "diversity": "Output diversity",
    "entropy_norm": "Output entropy",
    "drift_norm": "Distribution drift",
    "confidence": "Prediction confidence",
    "synthetic_ratio": "Synthetic data ratio",
    "accuracy": "Model accuracy",
}


class CollapseExplainer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.explainer = shap.TreeExplainer(classifier)

    def explain(self, features_row: dict, top_k=3):
        """
        Returns a list of (feature_name, shap_value, direction) for the
        predicted class, sorted by |impact|, plus a generated sentence.
        """
        X = pd.DataFrame([features_row])[FEATURE_COLUMNS]
        pred_class = self.classifier.predict(X)[0]
        class_idx = list(self.classifier.classes_).index(pred_class)

        shap_values = self.explainer.shap_values(X)
        # shap_values shape handling across shap versions: list[per-class] or 3D array
        if isinstance(shap_values, list):
            vals = shap_values[class_idx][0]
        else:
            vals = np.asarray(shap_values)[0, :, class_idx]

        contributions = list(zip(FEATURE_COLUMNS, vals))
        contributions.sort(key=lambda t: abs(t[1]), reverse=True)
        top = contributions[:top_k]

        bullets = []
        for feat, val in top:
            direction = "is pushing risk UP" if val > 0 else "is helping keep risk DOWN"
            bullets.append(f"{FRIENDLY_NAMES.get(feat, feat)} {direction} "
                            f"(impact {val:+.3f})")

        sentence = (
            f"Predicted risk level: {pred_class}. Main drivers -> "
            + "; ".join(bullets) + "."
        )
        return {
            "predicted_risk": pred_class,
            "top_features": top,
            "explanation": sentence,
        }


if __name__ == "__main__":
    from predictive_engine import load_artifacts

    clf, _ = load_artifacts()
    explainer = CollapseExplainer(clf)

    demo_row = {
        "diversity": 0.42, "entropy_norm": 0.48, "drift_norm": 0.55,
        "confidence": 0.81, "synthetic_ratio": 0.62, "accuracy": 0.58,
    }
    result = explainer.explain(demo_row)
    print(result["explanation"])
