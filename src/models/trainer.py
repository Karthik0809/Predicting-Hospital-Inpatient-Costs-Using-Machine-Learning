"""Training orchestration: Optuna hyperparameter search + MLflow experiment tracking."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from src.config import settings
from src.data.loader import TARGET, load_data
from src.data.preprocessor import (
    build_pipeline,
    load_pipeline,
    prepare_features,
    save_pipeline,
)
from src.models.neural import NeuralNetRegressor
from src.models.traditional import build_model, default_params, evaluate, optuna_space

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent import safety
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False
    _MLFLOW_IMPORT_ERROR = exc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

TRADITIONAL_MODELS = ["ridge", "random_forest", "xgboost", "lightgbm"]
_METRICS_FILE = "metrics.json"
_BEST_MODEL_FILE = "best_model.joblib"
_BEST_MODEL_NAME_FILE = "best_model_name.txt"


# ── Public API ──────────────────────────────────────────────────────────────

def train_all(use_optuna: bool = True) -> dict[str, dict[str, float]]:
    """Full training pipeline. Returns test metrics for all models."""
    artifacts = settings.artifacts_path
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1. Load & split data
    df = load_data()
    X, y = prepare_features(df)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train,
        test_size=settings.val_size / (1 - settings.test_size),
        random_state=settings.random_state,
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(X_train_raw), len(X_val_raw), len(X_test_raw),
    )

    # 2. Fit & save preprocessing pipeline
    preprocessor = build_pipeline()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val   = preprocessor.transform(X_val_raw)
    X_test  = preprocessor.transform(X_test_raw)
    save_pipeline(preprocessor, artifacts)

    # 3. Setup MLflow (optional in deployment environments where mlflow may be incompatible)
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
    else:
        logger.warning(
            "MLflow is unavailable in this environment; continuing without experiment logging. "
            "Import error: %s",
            _MLFLOW_IMPORT_ERROR,
        )

    all_metrics: dict[str, dict[str, float]] = {}

    # 4. Train traditional models
    for name in TRADITIONAL_MODELS:
        logger.info("Training %s ...", name)
        if use_optuna:
            params = _tune_model(name, X_train, y_train.values, X_val, y_val.values)
        else:
            params = default_params(name, settings.random_state)

        model = build_model(name, params)
        model.fit(X_train, y_train.values)
        metrics = evaluate(y_test.values, model.predict(X_test))
        all_metrics[name] = metrics

        _log_to_mlflow(name, params, metrics, artifacts)
        joblib.dump(model, artifacts / f"{name}.joblib")
        logger.info("  %s | R²=%.4f | RMSE=%.0f | MAE=%.0f", name, metrics["r2"], metrics["rmse"], metrics["mae"])

    # 5. Train neural network
    logger.info("Training neural network ...")
    nn_params: dict[str, Any]
    if use_optuna:
        nn_params = _tune_neural(X_train, y_train.values, X_val, y_val.values)
    else:
        nn_params = {
            "hidden_dims": (256, 128, 64),
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "epochs": 50,
            "patience": 10,
        }
    nn = NeuralNetRegressor(**nn_params)
    nn.fit(X_train, y_train.values, X_val, y_val.values)
    nn_metrics = evaluate(y_test.values, nn.predict(X_test))
    all_metrics["neural_network"] = nn_metrics
    _log_to_mlflow("neural_network", nn_params, nn_metrics, artifacts)
    joblib.dump(nn, artifacts / "neural_network.joblib")
    logger.info(
        "  neural_network | R²=%.4f | RMSE=%.0f | MAE=%.0f",
        nn_metrics["r2"], nn_metrics["rmse"], nn_metrics["mae"],
    )

    # 6. Select & persist the best model
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["r2"])
    best_model = joblib.load(artifacts / f"{best_name}.joblib")
    joblib.dump(best_model, artifacts / _BEST_MODEL_FILE)
    (artifacts / _BEST_MODEL_NAME_FILE).write_text(best_name)
    logger.info("Best model: %s (R²=%.4f)", best_name, all_metrics[best_name]["r2"])

    # 7. Train quantile regression models (XGBoost) for proper prediction intervals
    logger.info("Training quantile regression models (10th / 90th percentile) ...")
    from xgboost import XGBRegressor as _XGB
    for alpha, tag in [(0.1, "xgb_q10"), (0.9, "xgb_q90")]:
        qm = _XGB(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cpu",
            random_state=settings.random_state,
            verbosity=0,
        )
        qm.fit(X_train, y_train.values)
        joblib.dump(qm, artifacts / f"{tag}.joblib")
    logger.info("Quantile models saved.")

    # 8. Save all individual model predictions on test set (for ensemble / comparison)
    test_preds: dict[str, list[float]] = {}
    for name in TRADITIONAL_MODELS:
        m = joblib.load(artifacts / f"{name}.joblib")
        test_preds[name] = m.predict(X_test).tolist()
    test_preds["neural_network"] = nn.predict(X_test).tolist()
    test_preds["y_true"] = y_test.values.tolist()
    (artifacts / "test_predictions.json").write_text(json.dumps(test_preds))

    # 9. Save metrics summary
    (artifacts / _METRICS_FILE).write_text(json.dumps(all_metrics, indent=2))
    return all_metrics


# ── Model loading helpers ───────────────────────────────────────────────────

def load_best_model(artifacts_dir: str | Path | None = None) -> tuple[Any, str]:
    """Return (model, model_name). Raises if artifacts are missing."""
    d = Path(artifacts_dir or settings.artifacts_dir)
    name = (d / _BEST_MODEL_NAME_FILE).read_text().strip()
    model = joblib.load(d / _BEST_MODEL_FILE)
    return model, name


def load_metrics(artifacts_dir: str | Path | None = None) -> dict[str, dict[str, float]]:
    d = Path(artifacts_dir or settings.artifacts_dir)
    return json.loads((d / _METRICS_FILE).read_text())


def artifacts_exist(artifacts_dir: str | Path | None = None) -> bool:
    d = Path(artifacts_dir or settings.artifacts_dir)
    return (
        (d / _BEST_MODEL_FILE).exists()
        and (d / "preprocessing_pipeline.joblib").exists()
    )


def load_quantile_models(artifacts_dir: str | Path | None = None) -> tuple[Any | None, Any | None]:
    """Return (q10_model, q90_model) or (None, None) if not yet trained."""
    d = Path(artifacts_dir or settings.artifacts_dir)
    q10_path, q90_path = d / "xgb_q10.joblib", d / "xgb_q90.joblib"
    if q10_path.exists() and q90_path.exists():
        return joblib.load(q10_path), joblib.load(q90_path)
    return None, None


def load_all_models(artifacts_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a dict of {model_name: model} for all trained models."""
    d = Path(artifacts_dir or settings.artifacts_dir)
    result: dict[str, Any] = {}
    for name in [*TRADITIONAL_MODELS, "neural_network"]:
        path = d / f"{name}.joblib"
        if path.exists():
            result[name] = joblib.load(path)
    return result


def load_test_predictions(artifacts_dir: str | Path | None = None) -> dict[str, list[float]]:
    d = Path(artifacts_dir or settings.artifacts_dir)
    path = d / "test_predictions.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ── Optuna internals ────────────────────────────────────────────────────────

def _tune_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = optuna_space(name, trial)
        model = build_model(name, params)
        model.fit(X_train, y_train)
        return -evaluate(y_val, model.predict(X_val))["r2"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=settings.n_trials, show_progress_bar=False)
    logger.info("  [Optuna] %s best R²=%.4f", name, -study.best_value)
    return optuna_space(name, study.best_trial)


def _tune_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = NeuralNetRegressor.optuna_params(trial)
        nn = NeuralNetRegressor(**params)
        nn.fit(X_train, y_train, X_val, y_val)
        return -evaluate(y_val, nn.predict(X_val))["r2"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=max(5, settings.n_trials // 3), show_progress_bar=False)
    logger.info("  [Optuna] neural_network best R²=%.4f", -study.best_value)
    return NeuralNetRegressor.optuna_params(study.best_trial)


# ── MLflow helper ────────────────────────────────────────────────────────────

def _log_to_mlflow(
    name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path,
) -> None:
    if not MLFLOW_AVAILABLE:
        return
    with mlflow.start_run(run_name=name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        model_path = artifacts_dir / f"{name}.joblib"
        if model_path.exists():
            mlflow.log_artifact(str(model_path))
