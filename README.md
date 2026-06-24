# Early Warning System for AI Model Collapse using Predictive Analytics

A working implementation of the project described in the proposal: a system
that monitors AI model health metrics, predicts the probability of model
collapse before it happens, explains *why* risk is rising, and recommends
corrective action.

## What's implemented

| Spec feature                     | Implementation |
|-----------------------------------|-----------------|
| AI Health Monitoring Score        | `src/feature_engineering.py::health_score` |
| Collapse Forecasting (7/30 day)   | `src/predictive_engine.py` (XGBoost classifiers) |
| Self-Healing Recommendation Engine| `src/recommendation_engine.py` |
| Explainable AI Collapse Analyzer  | `src/explainability.py` (SHAP) |
| Digital Twin simulation           | `src/digital_twin.py` |
| Dashboard & Alerts                | `dashboard/app.py` (Streamlit) |

Since no public "model collapse telemetry" dataset exists, `data/generate_data.py`
simulates 60 model lineages retrained over 60 generations under different
data-hygiene practices, with realistic degradation dynamics. This gives the
predictive engine a labeled dataset to learn from. **In a real deployment**,
replace this synthetic data with real per-generation metrics computed by
`feature_engineering.py` from your actual model's outputs (see the functions
`entropy_from_probs`, `diversity_from_samples`, `kl_divergence_drift`,
`confidence_score`, `synthetic_ratio`).

## Project structure

```
ai_collapse_ews/
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py              # simulates training data
│   └── simulated_model_generations.csv (generated)
├── src/
│   ├── feature_engineering.py        # entropy/diversity/drift/health score
│   ├── predictive_engine.py          # trains + serves risk & forecast models
│   ├── explainability.py             # SHAP-based explanations
│   ├── recommendation_engine.py      # rule-based corrective actions
│   └── digital_twin.py               # scenario simulation
├── models/                           # saved model artifacts (generated)
└── dashboard/
    └── app.py                        # Streamlit dashboard
```

## Execution steps

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the simulated monitoring dataset
python data/generate_data.py

# 4. Train the risk classifier + forecasting models
python src/predictive_engine.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

### Quick CLI checks (optional, no dashboard needed)

```bash
python src/explainability.py        # demo SHAP explanation
python src/recommendation_engine.py # demo recommended actions
python src/digital_twin.py          # demo do-nothing vs. intervene scenario
```

## Dashboard tabs

- **Live Monitor** — set the 6 live metrics with sliders, see the AI Health
  Score gauge, ML-predicted risk level, 0/7/30-day collapse forecast, SHAP
  explanation of the top risk drivers, and recommended corrective actions.
- **Lineage History** — pick any of the 60 simulated model lineages and see
  its full health trajectory across 60 generations.
- **Digital Twin** — project the current metrics forward N generations
  under "do nothing" vs. "inject real data" and compare outcomes.

## Extending to a real model

1. Wire `feature_engineering.py`'s metric functions into your model's
   inference/evaluation pipeline (e.g. compute them on a sample of outputs
   each retraining cycle).
2. Log the six resulting numbers per generation to a database/CSV in place
   of `simulated_model_generations.csv`.
3. Re-run `predictive_engine.py` periodically to retrain on real outcomes.
4. Point `dashboard/app.py`'s `DATA_PATH` at your live monitoring table.

## Notes on the academic write-up

The original proposal lists TensorFlow and an LSTM forecaster as optional
extensions — those are good "future work" items (e.g. an LSTM over raw
per-generation embedding sequences instead of hand-engineered features) and
are intentionally left out here to keep the reference implementation simple
and fully reproducible without GPU requirements. The Random Forest + XGBoost
combination already satisfies the "Predictive Analytics" and "Explainable
AI" requirements end-to-end.
