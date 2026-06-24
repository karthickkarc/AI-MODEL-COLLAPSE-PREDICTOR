"""
dashboard/app.py
------------------
Streamlit dashboard for the Early Warning System for AI Model Collapse.

Run with:
    streamlit run dashboard/app.py

Features:
  - Live AI Health Score gauge + risk level
  - Manual "what's my model's metrics right now" input panel
  - Historical risk trend chart for any simulated lineage
  - Collapse probability forecast table (Today / +7 days / +30 days)
  - SHAP-based explanation of the current risk prediction
  - Self-healing recommended actions
  - Digital Twin scenario comparison (do-nothing vs. intervene)
"""

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import (  # noqa: E402
    health_score, health_bucket, risk_level_from_health, FEATURE_COLUMNS
)
from predictive_engine import load_artifacts, predict_risk, predict_forecast  # noqa: E402
from explainability import CollapseExplainer  # noqa: E402
from recommendation_engine import recommend  # noqa: E402
from digital_twin import compare_scenarios  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_model_generations.csv")
RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

st.set_page_config(page_title="AI Model Collapse Early Warning System", layout="wide")


@st.cache_resource
def get_models():
    return load_artifacts()


@st.cache_data
def get_history():
    return pd.read_csv(DATA_PATH)


def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "AI Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#34495e"},
            "steps": [
                {"range": [0, 50], "color": "#e74c3c"},
                {"range": [50, 70], "color": "#f39c12"},
                {"range": [70, 90], "color": "#3498db"},
                {"range": [90, 100], "color": "#2ecc71"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.title("🛡️ Early Warning System for AI Model Collapse")
    st.caption("Predictive analytics for detecting model collapse before it happens.")

    clf, forecasters = get_models()
    history = get_history()

    tab_monitor, tab_history, tab_twin = st.tabs(
        ["📊 Live Monitor", "📈 Lineage History", "🧬 Digital Twin"]
    )

    # ----------------------------------------------------------------
    with tab_monitor:
        st.subheader("Current Model Metrics")
        st.caption(
            "In production, these six values are computed automatically from your "
            "model's live outputs via `src/feature_engineering.py`. Adjust them here "
            "to see how the system reacts."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            diversity = st.slider("Output diversity", 0.0, 1.0, 0.55, 0.01)
            entropy_norm = st.slider("Output entropy", 0.0, 1.0, 0.60, 0.01)
        with col2:
            drift_norm = st.slider("Distribution drift", 0.0, 1.0, 0.35, 0.01)
            confidence = st.slider("Prediction confidence", 0.0, 1.0, 0.72, 0.01)
        with col3:
            synthetic_ratio = st.slider("Synthetic data ratio", 0.0, 1.0, 0.45, 0.01)
            accuracy = st.slider("Model accuracy", 0.0, 1.0, 0.70, 0.01)

        features_row = {
            "diversity": diversity, "entropy_norm": entropy_norm,
            "drift_norm": drift_norm, "confidence": confidence,
            "synthetic_ratio": synthetic_ratio, "accuracy": accuracy,
        }

        h_score = health_score(diversity, entropy_norm, drift_norm, accuracy, synthetic_ratio)
        bucket = health_bucket(h_score)
        pred_risk, proba = predict_risk(clf, features_row)

        st.divider()
        g1, g2 = st.columns([1, 1.4])
        with g1:
            st.plotly_chart(gauge_chart(h_score), use_container_width=True)
            st.markdown(f"**Status:** {bucket}")
        with g2:
            st.markdown("##### Predicted Risk Level (ML classifier)")
            badge_color = RISK_COLORS[pred_risk]
            st.markdown(
                f"<h2 style='color:{badge_color}'>{pred_risk} Risk</h2>",
                unsafe_allow_html=True,
            )
            proba_df = pd.DataFrame({"Risk Level": list(proba.keys()),
                                      "Probability": [round(v, 3) for v in proba.values()]})
            st.bar_chart(proba_df.set_index("Risk Level"))

        st.divider()
        st.subheader("Collapse Probability Forecast")
        forecast = predict_forecast(forecasters, features_row)
        forecast_df = pd.DataFrame({
            "Horizon": ["Today", "+7 Days", "+30 Days"],
            "Collapse Probability (%)": [forecast[0], forecast[7], forecast[30]],
        })
        fc1, fc2 = st.columns([1, 1.2])
        with fc1:
            st.table(forecast_df.set_index("Horizon"))
        with fc2:
            fig = px.line(forecast_df, x="Horizon", y="Collapse Probability (%)",
                           markers=True, range_y=[0, 100])
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            st.subheader("🔍 Explainable Analysis (SHAP)")
            explainer = CollapseExplainer(clf)
            result = explainer.explain(features_row)
            for feat, val in result["top_features"]:
                arrow = "🔺" if val > 0 else "🔻"
                st.write(f"{arrow} **{feat}** — impact {val:+.3f}")
            st.info(result["explanation"])
        with e2:
            st.subheader("🩹 Recommended Actions")
            actions = recommend(features_row, pred_risk)
            for a in actions:
                st.write(f"- {a}")

    # ----------------------------------------------------------------
    with tab_history:
        st.subheader("Historical Risk Trend by Model Lineage")
        lineage_ids = sorted(history["lineage_id"].unique())
        chosen = st.selectbox("Select a model lineage", lineage_ids)
        sub = history[history["lineage_id"] == chosen]

        fig = px.line(sub, x="generation", y="health_score",
                       title=f"Lineage {chosen}: Health Score over Generations")
        fig.add_hline(y=90, line_dash="dot", line_color="green", annotation_text="Healthy")
        fig.add_hline(y=70, line_dash="dot", line_color="blue", annotation_text="Stable")
        fig.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Warning")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(sub, x="generation",
                        y=["diversity", "entropy_norm", "drift_norm", "synthetic_ratio", "accuracy"],
                        title="Underlying metrics over time")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(sub[["generation"] + FEATURE_COLUMNS + ["health_score", "risk_level"]],
                     use_container_width=True)

    # ----------------------------------------------------------------
    with tab_twin:
        st.subheader("Digital Twin: Simulate the Future")
        st.caption(
            "Project the *current* metrics (from the Live Monitor tab) forward under "
            "two scenarios: continuing as-is vs. intervening with real-data refreshes."
        )
        n_gen = st.slider("Generations to simulate forward", 5, 40, 20)

        do_nothing, intervene = compare_scenarios(features_row, n_generations=n_gen)
        do_nothing["scenario"] = "Do nothing"
        intervene["scenario"] = "Inject real data"
        combined = pd.concat([do_nothing, intervene])

        fig = px.line(combined, x="generation", y="health_score", color="scenario",
                       title="Projected Health Score: Do Nothing vs. Intervene")
        fig.add_hline(y=50, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Do-nothing outcome**")
            st.write(do_nothing.iloc[-1][["health_score", "risk_level"]])
        with c2:
            st.markdown("**Intervention outcome**")
            st.write(intervene.iloc[-1][["health_score", "risk_level"]])


if __name__ == "__main__":
    main()
