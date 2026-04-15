"""Traditional ML models: Linear Regression, Random Forest, XGBoost, LightGBM."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def build_model(name: str, params: dict[str, Any] | None = None) -> Any:
    """Factory: return a sklearn-compatible regressor by name."""
    params = params or {}
    models = {
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models)}")
    return models[name](**params)


def default_params(name: str, random_state: int = 42) -> dict[str, Any]:
    """Sensible defaults for each model (used when Optuna is skipped)."""
    defaults: dict[str, dict[str, Any]] = {
        "ridge": {"alpha": 1.0},
        "random_forest": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 4,
            "n_jobs": -1,
            "random_state": random_state,
        },
        "xgboost": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": "cpu",
            "random_state": random_state,
            "verbosity": 0,
        },
        "lightgbm": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": random_state,
            "verbose": -1,
        },
    }
    return defaults[name]


def optuna_space(name: str, trial: Any) -> dict[str, Any]:
    """Return Optuna-sampled hyperparameters for a given model name."""
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "n_jobs": -1,
            "random_state": 42,
        }

    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tree_method": "hist",
            "device": "cpu",
            "random_state": 42,
            "verbosity": 0,
        }

    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

    raise ValueError(f"No Optuna space defined for '{name}'")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
    }
