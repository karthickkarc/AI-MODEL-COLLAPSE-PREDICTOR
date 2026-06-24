"""
generate_data.py
-----------------
Simulates monitoring logs for many independent "model lineages", each
retrained over 60 generations on a growing mix of synthetic + real data.

This stands in for real production telemetry. In a live deployment, you
would replace this script's output with actual per-generation metrics
computed by `src/feature_engineering.py` from your model's real outputs.

Ground-truth degradation dynamics (used ONLY to build a realistic, labeled
training set -- the predictive engine never sees these hidden parameters,
only the resulting observable features):

  - Each lineage has a random "real_data_injection" practice (0=never adds
    fresh real data, 1=aggressively refreshes with real data every cycle).
  - synthetic_ratio creeps up each generation unless real data is injected.
  - diversity, entropy and accuracy decay as a function of synthetic_ratio.
  - drift (KL divergence vs. generation-0) accumulates similarly.
  - confidence (overconfidence is itself a collapse symptom) rises as
    diversity collapses.

Output: data/simulated_model_generations.csv
Columns: lineage_id, generation, diversity, entropy_norm, drift_norm,
         confidence, synthetic_ratio, accuracy, health_score, risk_level
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import health_score, risk_level_from_health  # noqa: E402

RNG = np.random.default_rng(42)

N_LINEAGES = 60
N_GENERATIONS = 60


def simulate_lineage(lineage_id):
    # Random "ops practice": how much real data this team injects each cycle
    real_injection_rate = RNG.beta(2, 2)  # 0 (bad practice) -> 1 (good practice)
    decay = RNG.uniform(0.015, 0.05)      # how fragile this model architecture is
    noise = 0.01

    diversity = 1.0
    entropy_norm = 1.0
    drift_norm = 0.0
    confidence = RNG.uniform(0.55, 0.65)
    syn_ratio = RNG.uniform(0.02, 0.08)
    accuracy = RNG.uniform(0.90, 0.95)

    rows = []
    for gen in range(N_GENERATIONS):
        # synthetic data accumulates unless real data is injected this cycle
        injected = RNG.random() < real_injection_rate
        syn_growth = RNG.uniform(0.01, 0.05) * (1 - (0.8 if injected else 0))
        syn_ratio = float(np.clip(syn_ratio + syn_growth, 0, 1))

        pressure = decay * syn_ratio

        diversity = float(np.clip(diversity - pressure * RNG.uniform(0.8, 1.2) + RNG.normal(0, noise), 0, 1))
        entropy_norm = float(np.clip(entropy_norm - pressure * RNG.uniform(0.6, 1.0) + RNG.normal(0, noise), 0, 1))
        drift_norm = float(np.clip(drift_norm + pressure * RNG.uniform(1.5, 2.5) + RNG.normal(0, noise), 0, 1))
        confidence = float(np.clip(confidence + pressure * RNG.uniform(0.3, 0.6) + RNG.normal(0, noise), 0, 1))
        accuracy = float(np.clip(accuracy - pressure * RNG.uniform(0.5, 0.9) + RNG.normal(0, noise), 0, 1))

        h_score = health_score(diversity, entropy_norm, drift_norm, accuracy, syn_ratio)
        risk = risk_level_from_health(h_score)

        rows.append({
            "lineage_id": lineage_id,
            "generation": gen,
            "diversity": diversity,
            "entropy_norm": entropy_norm,
            "drift_norm": drift_norm,
            "confidence": confidence,
            "synthetic_ratio": syn_ratio,
            "accuracy": accuracy,
            "health_score": h_score,
            "risk_level": risk,
        })
    return rows


def main():
    all_rows = []
    for lid in range(N_LINEAGES):
        all_rows.extend(simulate_lineage(lid))

    df = pd.DataFrame(all_rows)

    # Forward-looking labels used to train the forecasting model:
    # accuracy this lineage will have H generations from now (NaN if no future data)
    for horizon in (7, 30):
        df[f"future_accuracy_{horizon}"] = (
            df.groupby("lineage_id")["accuracy"].shift(-horizon)
        )
        # Binary collapse event: accuracy crosses below 0.5 within the horizon
        collapsed_within = (
            df.groupby("lineage_id")["accuracy"]
            .apply(lambda s: s.shift(-horizon).rolling(horizon, min_periods=1).min() < 0.5)
            .reset_index(drop=True)
        )
        df[f"collapse_event_{horizon}d"] = collapsed_within.astype("float")

    out_dir = os.path.join(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "simulated_model_generations.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {N_LINEAGES} lineages to {out_path}")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    main()
