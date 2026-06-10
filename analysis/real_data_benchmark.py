"""Real-data benchmark on a SPARCS API sample: SHAP, residuals, significance tests, MLflow.

Downloads (or reuses) a 150K-row sample of the real NY SPARCS 2012 dataset via the
public Socrata API, trains XGBoost, and produces:
  - assets/shap_feature_importance.png
  - assets/residual_analysis.png
  - Welch t-tests on the README cost-driver findings
  - an MLflow run (mlruns/) with params, metrics, and artifacts

Run from repo root:  python analysis/real_data_benchmark.py
"""
from __future__ import annotations

import os
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import mlflow

SAMPLE_CSV = "data_sample.csv"
# The API returns rows ordered by facility, so a single $limit slice is biased
# (first rows skew to a few hospitals). Pull spaced chunks across all 2.5M rows.
API_BASE = (
    "https://health.data.ny.gov/resource/u4ud-w55t.csv"
    "?$select=age_group,gender,length_of_stay,type_of_admission,"
    "apr_severity_of_illness_code,apr_severity_of_illness_description,"
    "apr_risk_of_mortality,apr_medical_surgical_description,"
    "apr_drg_code,apr_mdc_code,ccs_diagnosis_code,"
    "emergency_department_indicator,total_costs"
)
TOTAL_ROWS = 2_544_543
N_CHUNKS = 10
CHUNK_SIZE = 15_000

CAT_COLS = ["age_group", "gender", "type_of_admission", "apr_risk_of_mortality",
            "apr_medical_surgical_description", "emergency_department_indicator"]
NUM_COLS = ["length_of_stay", "apr_severity_of_illness_code", "apr_drg_code",
            "apr_mdc_code", "ccs_diagnosis_code"]


def load_sample() -> pd.DataFrame:
    if not os.path.exists(SAMPLE_CSV):
        print("Downloading 150K-row spaced sample from NY Open Health Data API ...")
        frames = []
        step = TOTAL_ROWS // N_CHUNKS
        for i in range(N_CHUNKS):
            offset = i * step
            url = f"{API_BASE}&$limit={CHUNK_SIZE}&$offset={offset}"
            frames.append(pd.read_csv(url))
            print(f"  chunk {i+1}/{N_CHUNKS} (offset {offset:,})")
        pd.concat(frames, ignore_index=True).to_csv(SAMPLE_CSV, index=False)
    df = pd.read_csv(SAMPLE_CSV)
    df["length_of_stay"] = pd.to_numeric(
        df["length_of_stay"].astype(str).str.replace(r"\D", "", regex=True), errors="coerce"
    )
    for c in ["apr_severity_of_illness_code", "apr_drg_code", "apr_mdc_code",
              "ccs_diagnosis_code", "total_costs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=NUM_COLS + ["total_costs"])
    return df


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("hospital-cost-real-sample")

    df = load_sample()
    print(f"Sample: {len(df):,} rows")

    # ── Significance tests on README cost-driver findings ─────────────────────
    print("\n=== Welch t-tests on cost-driver findings ===")
    sev = df["apr_severity_of_illness_description"]
    extreme = df.loc[sev == "Extreme", "total_costs"]
    minor = df.loc[sev == "Minor", "total_costs"]
    t1, p1 = ttest_ind(extreme, minor, equal_var=False)
    print(f"Severity Extreme vs Minor: ${extreme.mean():,.0f} vs ${minor.mean():,.0f}"
          f"  ({extreme.mean()/minor.mean():.1f}x)  t={t1:.1f}  p={p1:.2e}")

    adm = df["type_of_admission"]
    trauma = df.loc[adm == "Trauma", "total_costs"]
    emergency = df.loc[adm == "Emergency", "total_costs"]
    t2, p2 = ttest_ind(trauma, emergency, equal_var=False)
    print(f"Admission Trauma vs Emergency: ${trauma.mean():,.0f} vs ${emergency.mean():,.0f}"
          f"  ({trauma.mean()/emergency.mean():.2f}x)  t={t2:.1f}  p={p2:.2e}")

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    X = pd.get_dummies(df[CAT_COLS + NUM_COLS], columns=CAT_COLS)
    y = df["total_costs"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    params = dict(n_estimators=400, max_depth=7, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=42)

    with mlflow.start_run(run_name="XGBoost-real-150k-sample"):
        mlflow.log_params(params)
        mlflow.log_param("sample_rows", len(df))
        mlflow.log_param("data_source", "NY SPARCS 2012 via Socrata API")

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        print(f"\nXGBoost on real 150K sample:  R2={r2:.3f}  RMSE=${rmse:,.0f}  MAE=${mae:,.0f}")

        # ── SHAP feature importance ────────────────────────────────────────────
        import shap

        explainer = shap.TreeExplainer(model)
        bg = X_te.sample(n=min(5000, len(X_te)), random_state=42)
        sv = explainer.shap_values(bg)
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values()

        fig, ax = plt.subplots(figsize=(9, 6))
        mean_abs.tail(15).plot.barh(ax=ax, color="#2563eb")
        ax.set_title("SHAP Feature Importance — XGBoost on 150K real SPARCS records")
        ax.set_xlabel("mean(|SHAP value|)  (impact on predicted cost, $)")
        plt.tight_layout()
        plt.savefig("assets/shap_feature_importance.png", dpi=130, bbox_inches="tight")
        mlflow.log_artifact("assets/shap_feature_importance.png")
        plt.close()
        print("Top 5 features:", list(mean_abs.tail(5).index[::-1]))

        # ── Residual analysis ──────────────────────────────────────────────────
        residuals = y_te - pred
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        hb = axes[0].hexbin(pred, residuals, gridsize=60, bins="log", cmap="Blues",
                            extent=(0, 60000, -40000, 40000))
        axes[0].axhline(0, color="#d62728", lw=1)
        axes[0].set_xlabel("Predicted cost ($)")
        axes[0].set_ylabel("Residual (actual - predicted, $)")
        axes[0].set_title("Residuals vs predicted (log-density)")
        fig.colorbar(hb, ax=axes[0], label="log10(count)")

        pct_err = residuals / np.maximum(y_te, 1) * 100
        bands = pd.cut(y_te, [0, 5000, 10000, 20000, 50000, np.inf],
                       labels=["<5K", "5-10K", "10-20K", "20-50K", ">50K"])
        band_mae = pd.Series(np.abs(residuals)).groupby(bands, observed=True).mean()
        band_mae.plot.bar(ax=axes[1], color="#2563eb")
        axes[1].set_xlabel("Actual cost band ($)")
        axes[1].set_ylabel("MAE ($)")
        axes[1].set_title("Where the model fails: error by cost band")
        plt.tight_layout()
        plt.savefig("assets/residual_analysis.png", dpi=130, bbox_inches="tight")
        mlflow.log_artifact("assets/residual_analysis.png")
        plt.close()
        print("MAE by cost band:")
        print(band_mae.round(0).to_string())
        print(f"\nUnder-prediction rate on stays >$50K: "
              f"{float((residuals[y_te > 50000] > 0).mean()) * 100:.0f}%")


if __name__ == "__main__":
    main()
