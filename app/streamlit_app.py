"""Hospital Inpatient Cost Predictor — Streamlit Dashboard v2.

Tabs
----
1. Overview      — project summary, model comparison, live KPIs
2. Predict       — single-patient prediction + SHAP waterfall + risk badge + proper CI
3. What-If       — interactive sensitivity sliders + tornado chart
4. Batch         — CSV upload → bulk predictions + risk distribution + download
5. Insights      — full EDA: distributions, correlations, feature importance
6. History       — session prediction log with CSV export
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .risk-low    {background:#1a5c38;color:#7dffb3;padding:6px 16px;border-radius:20px;font-weight:700;}
    .risk-medium {background:#7d5a00;color:#ffd980;padding:6px 16px;border-radius:20px;font-weight:700;}
    .risk-high   {background:#7d1a1a;color:#ff9090;padding:6px 16px;border-radius:20px;font-weight:700;}
    .predict-box {
        background:linear-gradient(135deg,#11998e,#38ef7d);
        border-radius:16px;padding:28px;color:#fff;text-align:center;
    }
    .stButton>button{border-radius:8px;font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
from src.data.loader import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.config import settings

SEVERITY_OPTS   = ["Minor", "Moderate", "Major", "Extreme"]
MORTALITY_OPTS  = ["Minor", "Moderate", "Major", "Extreme"]
ADMISSION_OPTS  = ["Emergency", "Urgent", "Elective", "Newborn", "Trauma", "Not Available"]
MED_SURG_OPTS   = ["Medical", "Surgical", "Not Applicable"]
PAYMENT_OPTS    = ["Medicare", "Medicaid", "Blue Cross/Blue Shield",
                   "Private Health Insurance", "Self-Pay",
                   "Miscellaneous/Other", "Federal/State/Local/VA"]
AREA_OPTS       = ["New York City", "Long Island", "Hudson Valley", "Capital/Adiron",
                   "Western NY", "Central NY", "Finger Lakes", "Southern Tier"]

# Feature ranges used for tornado / sensitivity chart
SENSITIVITY_RANGES: dict[str, list] = {
    "Length of Stay":             [1, 3, 5, 7, 10, 14, 21, 30],
    "APR Severity of Illness Code": [1, 2, 3, 4],
    "CCS Diagnosis Code":         [10, 50, 100, 150, 200, 250],
    "APR DRG Code":               [50, 200, 400, 600, 800, 940],
    "APR MDC Code":               [0, 5, 10, 15, 20, 25],
    "Birth Weight":               [0, 1000, 2000, 3000, 4000],
}


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_artifacts():
    from src.data.preprocessor import load_pipeline
    from src.models.trainer import (
        artifacts_exist, load_best_model, load_metrics, train_all,
    )
    if not artifacts_exist():
        train_all(use_optuna=False)
    model, name    = load_best_model()
    preprocessor   = load_pipeline(settings.artifacts_dir)
    metrics        = load_metrics()
    return model, name, preprocessor, metrics


@st.cache_resource(show_spinner=False)
def _get_support():
    """Load quantile models, all-model dict, and test predictions."""
    from src.models.trainer import (
        load_all_models, load_quantile_models, load_test_predictions,
    )
    q10, q90   = load_quantile_models()
    all_models = load_all_models()
    test_preds = load_test_predictions()
    return q10, q90, all_models, test_preds


@st.cache_resource(show_spinner=False)
def _get_explainer():
    from src.models.explainer import CostExplainer
    model, *_ = _get_artifacts()
    return CostExplainer(model, ALL_FEATURES)


@st.cache_data(show_spinner=False)
def _sample_data(n: int = 5_000) -> pd.DataFrame:
    from src.data.loader import load_data
    return load_data(sample_size=n)


# ── Bootstrap ────────────────────────────────────────────────────────────────
if "artifacts_loaded" not in st.session_state:
    with st.spinner("First-time setup: training all models (~90 s on free tier)…"):
        model, model_name, preprocessor, all_metrics = _get_artifacts()
    st.session_state["artifacts_loaded"] = True
else:
    model, model_name, preprocessor, all_metrics = _get_artifacts()

q10_model, q90_model, all_models, test_preds = _get_support()
explainer = _get_explainer()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _predict_cost(row: dict, mdl=None) -> float:
    mdl = mdl or model
    df  = pd.DataFrame([row])
    X   = preprocessor.transform(df[ALL_FEATURES])
    return float(max(mdl.predict(X)[0], 0.0))


def _predict_ci(row: dict) -> tuple[float, float]:
    """Return (lower, upper) 80th-percentile CI using quantile models."""
    df = pd.DataFrame([row])
    X  = preprocessor.transform(df[ALL_FEATURES])
    lo = float(max(q10_model.predict(X)[0], 0.0)) if q10_model else 0.0
    hi = float(max(q90_model.predict(X)[0], 0.0)) if q90_model else 0.0
    return lo, hi


def _risk_badge(cost: float) -> str:
    if cost < 10_000:
        return '<span class="risk-low">🟢 Low Risk</span>'
    if cost < 30_000:
        return '<span class="risk-medium">🟡 Medium Risk</span>'
    return '<span class="risk-high">🔴 High Risk</span>'


def _build_row(
    age_group, gender, los, severity_desc, severity_code,
    mortality_risk, admission_type, med_surg, payment,
    health_area, county, birth_weight, ccs_code, drg_code, mdc_code,
) -> dict:
    return {
        "Age Group": age_group,
        "Gender": gender,
        "Race": "Unknown",
        "Ethnicity": "Unknown",
        "Length of Stay": los,
        "Type of Admission": admission_type,
        "APR Severity of Illness Code": severity_code,
        "APR Severity of Illness Description": severity_desc,
        "APR Risk of Mortality": mortality_risk,
        "APR Medical Surgical Description": med_surg,
        "Payment Typology 1": payment,
        "Health Service Area": health_area,
        "Hospital County": county,
        "Birth Weight": birth_weight,
        "CCS Diagnosis Code": ccs_code,
        "APR DRG Code": drg_code,
        "APR MDC Code": mdc_code,
    }


def _sensitivity_df(row: dict) -> tuple[pd.DataFrame, float]:
    base = _predict_cost(row)
    rows = []
    for feat, values in SENSITIVITY_RANGES.items():
        costs = []
        for v in values:
            r = {**row, feat: v}
            costs.append(_predict_cost(r))
        rows.append({
            "Feature": feat,
            "Min Cost": min(costs),
            "Max Cost": max(costs),
            "Range":    max(costs) - min(costs),
        })
    df = pd.DataFrame(rows).sort_values("Range", ascending=True)
    return df, base


def _shap_waterfall_fig(shap_df: pd.DataFrame, base: float, pred: float) -> go.Figure:
    colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in shap_df["SHAP Value"]]
    df = shap_df.sort_values("SHAP Value", key=abs, ascending=True)
    fig = go.Figure(go.Bar(
        x=df["SHAP Value"], y=df["Feature"], orientation="h",
        marker_color=[
            "#e74c3c" if v >= 0 else "#27ae60" for v in df["SHAP Value"]
        ],
        text=[f"${v:+,.0f}" for v in df["SHAP Value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Feature Impact — Base ${base:,.0f} → Prediction ${pred:,.0f}",
        xaxis_title="Cost Impact (USD)",
        height=420, margin=dict(r=110),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _tornado_fig(sens_df: pd.DataFrame, base: float) -> go.Figure:
    fig = go.Figure()
    for _, row in sens_df.iterrows():
        lo_delta = row["Min Cost"] - base
        hi_delta = row["Max Cost"] - base
        fig.add_trace(go.Bar(
            x=[hi_delta - lo_delta],
            y=[row["Feature"]],
            base=lo_delta,
            orientation="h",
            marker_color="#667eea",
            text=f"  ±${row['Range']/2:,.0f}",
            textposition="outside",
            showlegend=False,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="white",
                  annotation_text="Baseline", annotation_position="top right")
    fig.update_layout(
        title=f"Sensitivity — Baseline ${base:,.0f}",
        xaxis_title="Cost vs Baseline ($)",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar — patient input ───────────────────────────────────────────────────
st.sidebar.title("🏥 Patient Details")
st.sidebar.caption("Configure a patient and click **Predict**.")

with st.sidebar.form("patient_form"):
    age_group  = st.selectbox("Age Group", ["0 to 17","18 to 29","30 to 49","50 to 69","70 or Older"], index=3)
    gender     = st.selectbox("Gender", ["M","F","U"])
    los        = st.number_input("Length of Stay (days)", 1, 365, 5)

    c1, c2 = st.columns(2)
    with c1:
        sev_desc  = st.selectbox("Severity", SEVERITY_OPTS, index=1)
        sev_code  = SEVERITY_OPTS.index(sev_desc) + 1
    with c2:
        mortality = st.selectbox("Mortality Risk", MORTALITY_OPTS, index=1)

    admission  = st.selectbox("Admission Type", ADMISSION_OPTS)
    med_surg   = st.selectbox("Case Type", MED_SURG_OPTS)
    payment    = st.selectbox("Payment", PAYMENT_OPTS)
    area       = st.selectbox("Health Service Area", AREA_OPTS)
    county     = st.text_input("Hospital County", "New York")
    bw         = st.number_input("Birth Weight (g) — 0 if not newborn", 0, 5000, 0)
    ccs        = st.slider("CCS Diagnosis Code", 1, 260, 100)
    drg        = st.slider("APR DRG Code", 1, 950, 300)
    mdc        = st.slider("APR MDC Code", 0, 25, 5)
    submitted  = st.form_submit_button("🔮  Predict Cost", use_container_width=True)

st.sidebar.divider()
if st.sidebar.button("🔄 Retrain Models", use_container_width=True):
    st.cache_resource.clear()
    st.cache_data.clear()
    for k in ["artifacts_loaded","history","last_row"]: st.session_state.pop(k, None)
    st.rerun()

# Always compute current_row so What-If / Batch can use defaults
current_row = _build_row(
    age_group, gender, los, sev_desc, sev_code,
    mortality, admission, med_surg, payment,
    area, county, bw, ccs, drg, mdc,
)
if submitted:
    st.session_state["last_row"] = current_row
    st.session_state.setdefault("history", [])


# ── Main title ────────────────────────────────────────────────────────────────
st.title("🏥 Hospital Inpatient Cost Predictor")
st.caption(
    f"Best model: **{model_name}** · "
    "Models: XGBoost · LightGBM · Random Forest · PyTorch MLP · "
    "Hyperparameter-tuned with Optuna · Tracked with MLflow"
)

metrics_df = (
    pd.DataFrame(all_metrics).T
    .reset_index().rename(columns={"index": "Model"})
    .sort_values("r2", ascending=False)
)

kpi_cols = st.columns(len(all_metrics))
for col, (_, row) in zip(kpi_cols, metrics_df.iterrows()):
    badge = " ⭐" if row["Model"] == model_name else ""
    col.metric(
        label=f"{row['Model']}{badge}",
        value=f"R² {row['r2']:.4f}",
        delta=f"RMSE ${row['rmse']:,.0f}",
        delta_color="inverse",
    )

(tab_ov, tab_pr, tab_wi, tab_bt, tab_ins, tab_hist) = st.tabs([
    "📊 Overview", "🔮 Predict", "🔧 What-If",
    "📁 Batch", "📈 Insights", "📋 History",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_ov:
    st.subheader("Model Performance Comparison")

    c_a, c_b = st.columns(2)
    with c_a:
        fig = px.bar(
            metrics_df, x="Model", y="r2",
            color="r2", color_continuous_scale="viridis",
            title="R² Score (higher is better)", text_auto=".4f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        fig = px.bar(
            metrics_df, x="Model", y="rmse",
            color="rmse", color_continuous_scale="reds_r",
            title="RMSE — USD (lower is better)", text_auto=",.0f",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    cats = ["r2_norm", "rmse_norm", "mae_norm"]
    radar_df = metrics_df.copy()
    radar_df["r2_norm"]   = radar_df["r2"]
    radar_df["rmse_norm"] = 1 - (radar_df["rmse"] / radar_df["rmse"].max())
    radar_df["mae_norm"]  = 1 - (radar_df["mae"]  / radar_df["mae"].max())

    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["r2_norm"], row["rmse_norm"], row["mae_norm"], row["r2_norm"]],
            theta=["R²", "RMSE (inv)", "MAE (inv)", "R²"],
            fill="toself", name=row["Model"],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Radar Comparison",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Full metrics table
    st.markdown("#### Full Metrics Table")
    st.dataframe(
        metrics_df.set_index("Model")
        .style
        .highlight_max(subset=["r2"], color="#1a5c38")
        .highlight_min(subset=["rmse", "mae"], color="#1a5c38")
        .format({"r2": "{:.4f}", "rmse": "${:,.0f}", "mae": "${:,.0f}", "mse": "{:,.0f}"}),
        use_container_width=True,
    )

    # If test predictions available, show actual vs predicted scatter
    if test_preds and "y_true" in test_preds:
        st.markdown("#### Actual vs Predicted — All Models")
        y_true = test_preds["y_true"]
        scatter_frames = []
        for name in all_metrics:
            if name in test_preds:
                scatter_frames.append(
                    pd.DataFrame({"Actual": y_true, "Predicted": test_preds[name], "Model": name})
                )
        if scatter_frames:
            sc_df = pd.concat(scatter_frames)
            fig_sc = px.scatter(
                sc_df.sample(min(500, len(sc_df))),
                x="Actual", y="Predicted", color="Model",
                opacity=0.5,
                title="Actual vs Predicted Cost (test set sample)",
                labels={"Actual": "Actual Cost ($)", "Predicted": "Predicted Cost ($)"},
            )
            # Perfect-prediction line
            mx = sc_df["Actual"].max()
            fig_sc.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines",
                line=dict(dash="dash", color="white"), name="Perfect Prediction",
            ))
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Tech Stack")
    st.markdown("""
| Layer | Technology |
|-------|-----------|
| ML Models | **XGBoost 2.x** · **LightGBM 4.x** · Random Forest · Ridge |
| Deep Learning | **PyTorch 2.x** MLP (BatchNorm + Dropout + AdamW + CosineAnnealing) |
| Hyperparameter Tuning | **Optuna 4.x** |
| Explainability | **SHAP** TreeExplainer |
| Prediction Intervals | **Quantile Regression** (XGBoost 10th/90th percentile) |
| Experiment Tracking | **MLflow 2.x** |
| Preprocessing | sklearn **ColumnTransformer Pipeline** |
| API | **FastAPI 0.115** |
| Config | **Pydantic-Settings v2** |
""")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_pr:
    if submitted or "last_row" in st.session_state:
        row = st.session_state.get("last_row", current_row)
        pred = _predict_cost(row)
        ci_lo, ci_hi = _predict_ci(row)

        # Percentile vs sample distribution
        sample_df = _sample_data(3_000)
        X_sample  = preprocessor.transform(sample_df[ALL_FEATURES])
        dist_preds = model.predict(X_sample)
        pctile = float(np.mean(dist_preds <= pred) * 100)

        # ── Result card ──────────────────────────────────────────────────────
        col_pred, col_meta = st.columns([2, 1])
        with col_pred:
            st.markdown(
                f"""
                <div class="predict-box">
                  <h2 style="margin:0 0 8px">Estimated Total Cost</h2>
                  <h1 style="font-size:3.2rem;font-weight:800;margin:0">${pred:,.2f}</h1>
                  <p style="opacity:.85;margin:6px 0 0">
                    80% CI: ${ci_lo:,.0f} — ${ci_hi:,.0f}
                  </p>
                  <p style="opacity:.7;font-size:.85rem;margin:4px 0 0">Model: {model_name}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_meta:
            st.markdown(f"**Risk Category** &nbsp; {_risk_badge(pred)}", unsafe_allow_html=True)
            st.metric("Daily Cost Estimate", f"${pred / max(row['Length of Stay'],1):,.0f}/day")
            st.metric("Cost Percentile", f"{pctile:.0f}th",
                      help="Compared to 3,000 sampled patients")
            st.metric("vs. Median Patient",
                      f"${pred - float(np.median(dist_preds)):+,.0f}",
                      delta_color="inverse")

        # ── All-model comparison ─────────────────────────────────────────────
        st.markdown("#### All-Model Predictions")
        model_preds = {}
        for mname, mdl in all_models.items():
            try:
                model_preds[mname] = _predict_cost(row, mdl)
            except Exception:
                pass
        if model_preds:
            mp_df = pd.DataFrame(
                {"Model": list(model_preds), "Prediction": list(model_preds.values())}
            ).sort_values("Prediction")
            fig_mp = px.bar(
                mp_df, x="Model", y="Prediction",
                title="Prediction from Each Model",
                color="Prediction", color_continuous_scale="viridis",
                text_auto="$,.0f",
            )
            fig_mp.update_layout(coloraxis_showscale=False)
            ensemble = float(np.mean(list(model_preds.values())))
            fig_mp.add_hline(
                y=ensemble, line_dash="dot", line_color="white",
                annotation_text=f"Ensemble avg ${ensemble:,.0f}",
            )
            st.plotly_chart(fig_mp, use_container_width=True)

        # ── SHAP waterfall ───────────────────────────────────────────────────
        if explainer.available:
            st.markdown("#### Why this prediction? (SHAP Feature Impact)")
            df_row = pd.DataFrame([row])
            X_row  = preprocessor.transform(df_row[ALL_FEATURES])
            shap_df = explainer.waterfall_data(X_row)
            if not shap_df.empty:
                st.plotly_chart(
                    _shap_waterfall_fig(shap_df, explainer.expected_value(), pred),
                    use_container_width=True,
                )
                st.caption(
                    "🔴 Red = feature pushes cost **up** · 🟢 Green = feature pushes cost **down** · "
                    "Bar length = magnitude of impact."
                )
        else:
            st.info("SHAP explainability not available for this model type.")

        # ── Cost distribution context ────────────────────────────────────────
        st.markdown("#### How does this compare to all patients?")
        fig_dist = px.histogram(
            dist_preds, nbins=50,
            title="Predicted Cost Distribution (sample of 3,000 patients)",
            labels={"value": "Predicted Cost ($)", "count": "# Patients"},
            color_discrete_sequence=["#667eea"],
        )
        fig_dist.add_vline(
            x=pred, line_dash="dash", line_color="#38ef7d",
            annotation_text=f"This patient ${pred:,.0f}",
            annotation_position="top right",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── Download ─────────────────────────────────────────────────────────
        out_df = pd.DataFrame([row]).assign(
            **{"Predicted Cost": pred, "CI Lower": ci_lo, "CI Upper": ci_hi,
               "Risk": "Low" if pred < 10_000 else ("Medium" if pred < 30_000 else "High"),
               "Model": model_name}
        )
        st.download_button(
            "⬇️ Download this prediction",
            data=out_df.to_csv(index=False).encode(),
            file_name="prediction.csv", mime="text/csv",
        )
        # Save to history
        st.session_state.setdefault("history", []).append({
            **{k: v for k, v in row.items() if k in ["Age Group","Gender","Length of Stay",
                                                       "Type of Admission",
                                                       "APR Severity of Illness Description"]},
            "Predicted Cost": f"${pred:,.2f}",
            "Risk": "Low" if pred < 10_000 else ("Medium" if pred < 30_000 else "High"),
            "Model": model_name,
        })
        st.session_state["last_row"] = row
    else:
        st.info("Configure patient details in the sidebar and click **Predict Cost**.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab_wi:
    st.subheader("🔧 What-If Simulator")
    st.caption(
        "Adjust parameters below to see how changing a single variable impacts cost. "
        "Tornado chart shows which features have the **largest** cost range."
    )

    base_row = st.session_state.get("last_row", current_row)
    base_cost = _predict_cost(base_row)

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        st.markdown("**Modify parameters:**")
        wi_los = st.slider(
            "Length of Stay",
            1, 30, int(base_row["Length of Stay"]),
            key="wi_los",
        )
        wi_sev_desc = st.select_slider(
            "Severity",
            options=SEVERITY_OPTS,
            value=base_row["APR Severity of Illness Description"],
            key="wi_sev",
        )
        wi_sev_code = SEVERITY_OPTS.index(wi_sev_desc) + 1

        wi_admission = st.selectbox(
            "Admission Type",
            ADMISSION_OPTS,
            index=ADMISSION_OPTS.index(base_row["Type of Admission"]),
            key="wi_adm",
        )
        wi_med_surg = st.selectbox(
            "Case Type",
            MED_SURG_OPTS,
            index=MED_SURG_OPTS.index(base_row["APR Medical Surgical Description"]),
            key="wi_ms",
        )
        wi_payment = st.selectbox(
            "Payment Type",
            PAYMENT_OPTS,
            index=PAYMENT_OPTS.index(base_row["Payment Typology 1"])
            if base_row["Payment Typology 1"] in PAYMENT_OPTS else 0,
            key="wi_pay",
        )

    # Build modified row
    wi_row = {
        **base_row,
        "Length of Stay": wi_los,
        "APR Severity of Illness Code": wi_sev_code,
        "APR Severity of Illness Description": wi_sev_desc,
        "Type of Admission": wi_admission,
        "APR Medical Surgical Description": wi_med_surg,
        "Payment Typology 1": wi_payment,
    }
    wi_cost  = _predict_cost(wi_row)
    delta    = wi_cost - base_cost
    delta_pc = delta / base_cost * 100 if base_cost else 0

    with col_result:
        st.markdown("**Cost impact:**")
        m1, m2 = st.columns(2)
        m1.metric("Baseline", f"${base_cost:,.0f}")
        m2.metric("Modified", f"${wi_cost:,.0f}",
                  delta=f"${delta:+,.0f}  ({delta_pc:+.1f}%)",
                  delta_color="inverse")

        fig_cmp = go.Figure(go.Bar(
            x=["Baseline", "Modified"],
            y=[base_cost, wi_cost],
            marker_color=[
                "#667eea",
                "#e74c3c" if delta > 0 else "#27ae60",
            ],
            text=[f"${base_cost:,.0f}", f"${wi_cost:,.0f}"],
            textposition="outside",
        ))
        fig_cmp.update_layout(
            title="Cost Comparison",
            yaxis_title="USD",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Tornado chart ─────────────────────────────────────────────────────────
    st.markdown("#### Sensitivity Tornado — Cost Range per Feature")
    st.caption(
        "Each bar shows the full range of predicted cost as that feature varies "
        "from its minimum to maximum value (all others held constant)."
    )
    with st.spinner("Computing sensitivity…"):
        sens_df, base_cost_for_tornado = _sensitivity_df(base_row)
    st.plotly_chart(_tornado_fig(sens_df, base_cost_for_tornado), use_container_width=True)

    # ── LOS sweep chart ───────────────────────────────────────────────────────
    st.markdown("#### Length of Stay vs Cost — sweep")
    los_costs = []
    for d in range(1, 31):
        los_costs.append({"Days": d, "Predicted Cost": _predict_cost({**base_row, "Length of Stay": d})})
    los_df = pd.DataFrame(los_costs)
    fig_los = px.line(
        los_df, x="Days", y="Predicted Cost",
        title="Cost vs Length of Stay (all else equal)",
        markers=True, color_discrete_sequence=["#38ef7d"],
    )
    fig_los.add_vline(
        x=base_row["Length of Stay"], line_dash="dash",
        line_color="white", annotation_text="Current",
    )
    st.plotly_chart(fig_los, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.subheader("📁 Batch Prediction")
    st.caption("Upload a CSV with patient records. Download results with predicted costs and risk categories.")

    # Template download
    template_row = {col: current_row.get(col, "") for col in ALL_FEATURES}
    template_df  = pd.DataFrame([template_row] * 3)
    st.download_button(
        "⬇️ Download template CSV",
        data=template_df.to_csv(index=False).encode(),
        file_name="batch_template.csv",
        mime="text/csv",
        help=f"Required columns: {', '.join(ALL_FEATURES)}",
    )

    uploaded = st.file_uploader(
        "Upload patient data CSV",
        type=["csv"],
        help="Must have the same columns as the template above.",
    )

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            missing  = [c for c in ALL_FEATURES if c not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()

            with st.spinner(f"Running predictions on {len(batch_df):,} rows…"):
                X_b   = preprocessor.transform(batch_df[ALL_FEATURES])
                preds = np.maximum(model.predict(X_b), 0.0)
                if q10_model and q90_model:
                    ci_lo_arr = np.maximum(q10_model.predict(X_b), 0.0)
                    ci_hi_arr = np.maximum(q90_model.predict(X_b), 0.0)
                else:
                    ci_lo_arr = preds * 0.85
                    ci_hi_arr = preds * 1.15

            batch_df["Predicted Cost"] = preds.round(2)
            batch_df["CI Lower"]       = ci_lo_arr.round(2)
            batch_df["CI Upper"]       = ci_hi_arr.round(2)
            batch_df["Risk Category"]  = pd.cut(
                preds,
                bins=[0, 10_000, 30_000, np.inf],
                labels=["Low", "Medium", "High"],
            )
            batch_df["Model"] = model_name

            # ── Summary KPIs ──────────────────────────────────────────────────
            st.success(f"✅ Predictions complete for **{len(batch_df):,}** patients.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Mean Cost",   f"${preds.mean():,.0f}")
            k2.metric("Median Cost", f"${np.median(preds):,.0f}")
            k3.metric("Max Cost",    f"${preds.max():,.0f}")
            k4.metric("Total Cost",  f"${preds.sum():,.0f}")

            # ── Charts ────────────────────────────────────────────────────────
            c_l, c_r = st.columns(2)
            with c_l:
                risk_c = batch_df["Risk Category"].value_counts().reset_index()
                risk_c.columns = ["Risk", "Count"]
                fig_pie = px.pie(
                    risk_c, values="Count", names="Risk",
                    title="Risk Category Distribution",
                    color="Risk",
                    color_discrete_map={"Low":"#27ae60","Medium":"#f39c12","High":"#e74c3c"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_r:
                fig_hist = px.histogram(
                    batch_df, x="Predicted Cost", nbins=40,
                    title="Predicted Cost Distribution",
                    color_discrete_sequence=["#667eea"],
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Results table ─────────────────────────────────────────────────
            st.markdown("#### Results Preview (first 100 rows)")
            display_cols = ["Predicted Cost", "CI Lower", "CI Upper", "Risk Category"] + ALL_FEATURES[:5]
            st.dataframe(batch_df[display_cols].head(100), use_container_width=True)

            st.download_button(
                "⬇️ Download full results (CSV)",
                data=batch_df.to_csv(index=False).encode(),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Error: {exc}")
    else:
        st.info("Download the template CSV, fill in your patient data, then upload it here.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS (EDA)
# ════════════════════════════════════════════════════════════════════════════
with tab_ins:
    st.subheader("📈 Data Insights & EDA")
    df_s = _sample_data(5_000)

    # Cost distribution
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            df_s, x="Total Costs", nbins=60,
            title="Total Cost Distribution",
            color_discrete_sequence=["#667eea"],
            marginal="box",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            np.log1p(df_s["Total Costs"]), nbins=60,
            title="Log(Total Costs) Distribution",
            color_discrete_sequence=["#38ef7d"],
            marginal="box",
            labels={"value": "log(1+Total Costs)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # By age & severity
    c3, c4 = st.columns(2)
    with c3:
        fig = px.violin(
            df_s, x="Age Group", y="Total Costs",
            color="Age Group", box=True,
            category_orders={"Age Group": ["0 to 17","18 to 29","30 to 49","50 to 69","70 or Older"]},
            title="Cost Distribution by Age Group",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(
            df_s, x="APR Severity of Illness Description", y="Total Costs",
            color="APR Severity of Illness Description",
            category_orders={"APR Severity of Illness Description": SEVERITY_OPTS},
            title="Cost by Severity of Illness",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # LOS scatter + payment bar
    c5, c6 = st.columns(2)
    with c5:
        samp = df_s.sample(min(2_000, len(df_s)))
        scatter_kwargs = dict(
            data_frame=samp,
            x="Length of Stay",
            y="Total Costs",
            color="APR Severity of Illness Description",
            opacity=0.45,
            title="Length of Stay vs Total Cost",
            category_orders={"APR Severity of Illness Description": SEVERITY_OPTS},
        )
        if importlib.util.find_spec("statsmodels") is not None:
            scatter_kwargs["trendline"] = "ols"
        fig = px.scatter(**scatter_kwargs)
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        pay_avg = (
            df_s.groupby("Payment Typology 1")["Total Costs"]
            .mean().reset_index().sort_values("Total Costs", ascending=False)
        )
        fig = px.bar(
            pay_avg, x="Total Costs", y="Payment Typology 1",
            orientation="h",
            title="Average Cost by Payment Type",
            color="Total Costs", color_continuous_scale="blues",
        )
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Admission type heatmap: admission x severity
    st.markdown("#### Average Cost Heatmap: Admission Type × Severity")
    hm = (
        df_s.groupby(["Type of Admission","APR Severity of Illness Description"])["Total Costs"]
        .mean().reset_index()
        .pivot(index="Type of Admission", columns="APR Severity of Illness Description", values="Total Costs")
        .reindex(columns=SEVERITY_OPTS)
    )
    fig_hm = px.imshow(
        hm, text_auto="$,.0f",
        color_continuous_scale="viridis",
        title="Mean Cost ($): Admission Type × Severity",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Correlation heatmap (numerical features)
    st.markdown("#### Correlation Matrix (Numerical Features)")
    num_cols_present = [c for c in NUMERICAL_FEATURES + ["Total Costs"] if c in df_s.columns]
    corr = df_s[num_cols_present].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # SHAP global importance
    if explainer.available:
        st.markdown("#### Global SHAP Feature Importance")
        X_imp = preprocessor.transform(df_s[ALL_FEATURES].head(500))
        gi_df = explainer.global_importance(X_imp)
        if not gi_df.empty:
            fig_gi = px.bar(
                gi_df.head(15), x="Importance", y="Feature",
                orientation="h", color="Importance",
                color_continuous_scale="viridis",
                title="Mean |SHAP| — Global Feature Impact (sample of 500)",
            )
            fig_gi.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_gi, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.subheader("📋 Prediction History (this session)")
    history = st.session_state.get("history", [])

    if not history:
        st.info("No predictions made yet. Go to the **Predict** tab to make one.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download history (CSV)",
                data=hist_df.to_csv(index=False).encode(),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

        # Mini trend chart
        if len(history) > 1:
            costs_raw = [
                float(h["Predicted Cost"].replace("$","").replace(",",""))
                for h in history if "Predicted Cost" in h
            ]
            fig_trend = px.line(
                pd.DataFrame({"Prediction #": range(1, len(costs_raw)+1), "Cost": costs_raw}),
                x="Prediction #", y="Cost",
                title="Cost Trend Across Predictions",
                markers=True, color_discrete_sequence=["#38ef7d"],
            )
            st.plotly_chart(fig_trend, use_container_width=True)
