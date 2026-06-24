"""
feature_engineering.py
-----------------------
Reusable functions to compute the core "vital sign" metrics used to monitor
AI model health over successive training generations:

    - Shannon entropy of output distribution
    - Diversity index (distinct-content ratio / effective vocabulary)
    - Distribution drift (KL divergence vs. a reference/original distribution)
    - Prediction confidence
    - Synthetic-to-real data ratio
    - Composite AI Health Score

These functions work on two kinds of input:
  1. Raw model output samples (e.g. token/label probability vectors) -> use
     `entropy_from_probs`, `diversity_from_samples`, `kl_divergence_drift`.
  2. Already-aggregated monitoring metrics (the common case once a pipeline
     is logging summary stats per generation) -> use `health_score` directly.
"""

import numpy as np
from collections import Counter


def entropy_from_probs(prob_vectors, base=2):
    """
    Shannon entropy averaged over a batch of probability vectors.

    prob_vectors: array-like, shape (n_samples, n_classes), rows sum to 1.
    Returns: average entropy (nats if base=np.e, bits if base=2).
    """
    prob_vectors = np.asarray(prob_vectors, dtype=float)
    prob_vectors = np.clip(prob_vectors, 1e-12, 1.0)
    ent = -np.sum(prob_vectors * np.log(prob_vectors), axis=1) / np.log(base)
    return float(np.mean(ent))


def diversity_from_samples(samples):
    """
    Simple diversity index: ratio of unique items to total items, e.g. for
    a batch of generated text outputs, sentence hashes, or class labels.
    1.0 = every sample unique (max diversity), -> 0 = total mode collapse.
    """
    if len(samples) == 0:
        return 0.0
    counts = Counter(samples)
    unique = len(counts)
    total = len(samples)
    return unique / total


def kl_divergence_drift(p, q, eps=1e-12):
    """
    KL(p || q): how far the *current* output distribution `p` has drifted
    from a *reference* distribution `q` (e.g. generation-0 distribution,
    or the real-world data distribution).
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def confidence_score(prob_vectors):
    """Average max-probability ('top-1 confidence') across a batch."""
    prob_vectors = np.asarray(prob_vectors, dtype=float)
    return float(np.mean(np.max(prob_vectors, axis=1)))


def synthetic_ratio(n_synthetic, n_real):
    total = n_synthetic + n_real
    if total == 0:
        return 0.0
    return n_synthetic / total


def health_score(diversity, entropy_norm, drift_norm, accuracy, synth_ratio,
                  weights=(0.30, 0.25, 0.20, 0.15, 0.10)):
    """
    Composite AI Health Score (0-100), per the design spec:

        Health = f(Diversity, Entropy, Drift, Accuracy, Confidence/Synthetic ratio)

    All inputs except drift_norm/synth_ratio are expected in [0,1] where
    HIGHER is healthier. drift_norm and synth_ratio are inverted internally
    (higher drift / higher synthetic ratio -> lower health).

    weights: (diversity, entropy, drift, accuracy, synthetic_ratio)
    """
    w_div, w_ent, w_drift, w_acc, w_syn = weights
    score = (
        w_div * diversity +
        w_ent * entropy_norm +
        w_drift * (1 - np.clip(drift_norm, 0, 1)) +
        w_acc * accuracy +
        w_syn * (1 - np.clip(synth_ratio, 0, 1))
    )
    return float(np.clip(score * 100, 0, 100))


def health_bucket(score):
    """Map a 0-100 health score to the categorical health status."""
    if score >= 90:
        return "Healthy"
    elif score >= 70:
        return "Stable"
    elif score >= 50:
        return "Warning"
    else:
        return "Collapse Risk"


def risk_level_from_health(score):
    """Map health score -> 3-level operational risk label."""
    bucket = health_bucket(score)
    if bucket in ("Healthy", "Stable"):
        return "Low"
    elif bucket == "Warning":
        return "Medium"
    else:
        return "High"


FEATURE_COLUMNS = [
    "diversity", "entropy_norm", "drift_norm",
    "confidence", "synthetic_ratio", "accuracy",
]
