"""
recommendation_engine.py
-------------------------
Rule-based "self-healing" recommendation engine. Given the current feature
vector and risk level, returns prioritized, concrete corrective actions.

Thresholds are deliberately simple/transparent (vs. a black-box model)
because recommendations need to be auditable and explainable to an ML ops
team -- this complements, rather than replaces, the SHAP explanations.
"""

RULES = [
    # (condition_fn, action, priority)
    (lambda f: f["synthetic_ratio"] > 0.5,
     "Reduce synthetic data usage — synthetic-to-real ratio exceeds 50%.", 1),
    (lambda f: f["diversity"] < 0.6,
     "Inject additional real-world data to restore output diversity "
     "(target: +20% real samples in next training batch).", 1),
    (lambda f: f["drift_norm"] > 0.4,
     "Retrain against a fresh real-data validation set — distribution has "
     "drifted significantly from the original reference.", 2),
    (lambda f: f["entropy_norm"] < 0.6,
     "Increase dataset diversity / sampling temperature — output entropy "
     "is collapsing toward repetitive generations.", 2),
    (lambda f: f["confidence"] > 0.85 and f["accuracy"] < 0.75,
     "Investigate overconfidence: high prediction confidence paired with "
     "falling accuracy is a classic early collapse signature.", 1),
    (lambda f: f["accuracy"] < 0.6,
     "Pause automated retraining on synthetic outputs and roll back to the "
     "last known-good checkpoint.", 1),
]

DEFAULT_OK = "No corrective action needed — all monitored metrics within healthy range."


def recommend(features_row: dict, risk_level: str = None):
    """
    Returns a list of recommended actions, sorted by priority (1=urgent).
    """
    triggered = []
    for condition, action, priority in RULES:
        try:
            if condition(features_row):
                triggered.append((priority, action))
        except KeyError:
            continue

    triggered.sort(key=lambda t: t[0])
    actions = [a for _, a in triggered]

    if not actions:
        actions = [DEFAULT_OK]

    return actions


if __name__ == "__main__":
    demo_row = {
        "diversity": 0.42, "entropy_norm": 0.48, "drift_norm": 0.55,
        "confidence": 0.81, "synthetic_ratio": 0.62, "accuracy": 0.58,
    }
    for action in recommend(demo_row, "High"):
        print("-", action)
