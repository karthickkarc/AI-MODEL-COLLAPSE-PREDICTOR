"""
digital_twin.py
-----------------
A lightweight "Digital Twin" of the model lineage: projects the current
feature vector forward N generations under a chosen intervention scenario
(e.g. "do nothing" vs. "inject 20% real data each cycle"), scoring each
projected generation with the trained predictive engine.

This lets a team simulate "what happens if we keep training on synthetic
data" vs. "what happens if we intervene now" *before* committing to either
path on the real model.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from feature_engineering import health_score, risk_level_from_health, FEATURE_COLUMNS  # noqa: E402


def project_generation(features_row: dict, real_data_injection: float, decay: float = 0.03):
    """
    Advance the feature vector by one simulated generation.

    real_data_injection: 0.0 (no real data added) to 1.0 (heavy real-data
                          refresh each cycle) -- mirrors the practice knob
                          used in data/generate_data.py.
    """
    f = dict(features_row)
    syn_growth = 0.03 * (1 - 0.8 * real_data_injection)
    f["synthetic_ratio"] = float(np.clip(f["synthetic_ratio"] + syn_growth, 0, 1))

    pressure = decay * f["synthetic_ratio"]
    f["diversity"] = float(np.clip(f["diversity"] - pressure, 0, 1))
    f["entropy_norm"] = float(np.clip(f["entropy_norm"] - pressure * 0.8, 0, 1))
    f["drift_norm"] = float(np.clip(f["drift_norm"] + pressure * 2.0, 0, 1))
    f["confidence"] = float(np.clip(f["confidence"] + pressure * 0.4, 0, 1))
    f["accuracy"] = float(np.clip(f["accuracy"] - pressure * 0.7, 0, 1))
    return f


def simulate_scenario(features_row: dict, n_generations: int, real_data_injection: float):
    """
    Returns a DataFrame with one row per simulated future generation:
    generation, health_score, risk_level, and all feature values.
    """
    state = dict(features_row)
    rows = []
    for gen in range(1, n_generations + 1):
        state = project_generation(state, real_data_injection)
        h = health_score(state["diversity"], state["entropy_norm"],
                          state["drift_norm"], state["accuracy"], state["synthetic_ratio"])
        rows.append({
            "generation": gen,
            **{k: state[k] for k in FEATURE_COLUMNS},
            "health_score": h,
            "risk_level": risk_level_from_health(h),
        })
    return pd.DataFrame(rows)


def compare_scenarios(features_row: dict, n_generations: int = 20):
    """Convenience helper: compare 'no intervention' vs. 'real-data refresh'."""
    do_nothing = simulate_scenario(features_row, n_generations, real_data_injection=0.0)
    intervene = simulate_scenario(features_row, n_generations, real_data_injection=0.8)
    return do_nothing, intervene


if __name__ == "__main__":
    demo_row = {
        "diversity": 0.75, "entropy_norm": 0.78, "drift_norm": 0.15,
        "confidence": 0.68, "synthetic_ratio": 0.30, "accuracy": 0.85,
    }
    nothing, intervene = compare_scenarios(demo_row, n_generations=15)
    print("Do-nothing scenario, final generation:")
    print(nothing.iloc[-1][["health_score", "risk_level"]])
    print("\nIntervention scenario, final generation:")
    print(intervene.iloc[-1][["health_score", "risk_level"]])
