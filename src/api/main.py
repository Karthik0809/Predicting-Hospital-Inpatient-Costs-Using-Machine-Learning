"""FastAPI application for hospital inpatient cost prediction."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    MetricsResponse,
    PatientRecord,
    PredictionResponse,
)
from src.config import settings
from src.data.loader import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.data.preprocessor import load_pipeline
from src.models.trainer import artifacts_exist, load_best_model, load_metrics

logger = logging.getLogger(__name__)

# ── App state ────────────────────────────────────────────────────────────────
_state: dict[str, Any] = {
    "model": None,
    "model_name": None,
    "preprocessor": None,
    "metrics": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup."""
    if artifacts_exist():
        try:
            _state["model"], _state["model_name"] = load_best_model()
            _state["preprocessor"] = load_pipeline(settings.artifacts_dir)
            _state["metrics"] = load_metrics()
            logger.info("Loaded model '%s' from artifacts.", _state["model_name"])
        except Exception as exc:
            logger.error("Failed to load artifacts: %s", exc)
    else:
        logger.warning(
            "No artifacts found at '%s'. Run `python train.py` first.", settings.artifacts_dir
        )
    yield
    # (cleanup on shutdown if needed)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hospital Inpatient Cost Predictor",
    description=(
        "Predict total inpatient hospital costs using XGBoost, LightGBM, "
        "Random Forest, and a PyTorch MLP. Trained on NY SPARCS data."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _record_to_df(record: PatientRecord) -> pd.DataFrame:
    row = {
        "Age Group": record.age_group,
        "Gender": record.gender,
        "Race": record.race,
        "Ethnicity": record.ethnicity,
        "Length of Stay": record.length_of_stay,
        "Type of Admission": record.type_of_admission,
        "APR Severity of Illness Code": record.apr_severity_code,
        "APR Severity of Illness Description": record.apr_severity_desc,
        "APR Risk of Mortality": record.apr_risk_of_mortality,
        "APR Medical Surgical Description": record.apr_medical_surgical,
        "Payment Typology 1": record.payment_typology,
        "Health Service Area": record.health_service_area,
        "Hospital County": record.hospital_county,
        "Birth Weight": record.birth_weight,
        "CCS Diagnosis Code": record.ccs_diagnosis_code,
        "APR DRG Code": record.apr_drg_code,
        "APR MDC Code": record.apr_mdc_code,
    }
    return pd.DataFrame([row])


def _require_artifacts() -> None:
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` to train and save artifacts.",
        )


def _predict_single(record: PatientRecord) -> float:
    df = _record_to_df(record)
    X = _state["preprocessor"].transform(df[ALL_FEATURES])
    pred = float(_state["model"].predict(X)[0])
    return max(pred, 0.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_name=_state["model_name"],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Status"])
async def get_metrics() -> MetricsResponse:
    _require_artifacts()
    return MetricsResponse(
        metrics=_state["metrics"],
        best_model=_state["model_name"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(record: PatientRecord) -> PredictionResponse:
    _require_artifacts()
    predicted = _predict_single(record)

    # Approximate 95% CI via ±15% (replace with quantile regression for production)
    ci_half = predicted * 0.15
    return PredictionResponse(
        predicted_cost=round(predicted, 2),
        model_used=_state["model_name"],
        confidence_interval_lower=round(max(predicted - ci_half, 0.0), 2),
        confidence_interval_upper=round(predicted + ci_half, 2),
        features_used=ALL_FEATURES,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(request: BatchRequest) -> BatchResponse:
    _require_artifacts()
    rows = [_record_to_df(r) for r in request.records]
    df = pd.concat(rows, ignore_index=True)
    X = _state["preprocessor"].transform(df[ALL_FEATURES])
    preds = _state["model"].predict(X).tolist()
    preds = [round(max(p, 0.0), 2) for p in preds]
    return BatchResponse(
        predictions=preds,
        model_used=_state["model_name"],
        count=len(preds),
    )


@app.get("/models", tags=["Status"])
async def list_models() -> dict[str, Any]:
    _require_artifacts()
    metrics = _state["metrics"] or {}
    return {
        "available_models": list(metrics.keys()),
        "best_model": _state["model_name"],
        "model_metrics": metrics,
    }


@app.get("/", tags=["Status"])
async def root() -> dict[str, str]:
    return {
        "message": "Hospital Inpatient Cost Predictor API v2.0",
        "docs": "/docs",
        "health": "/health",
    }
