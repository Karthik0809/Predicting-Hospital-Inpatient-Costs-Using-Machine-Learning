"""Scikit-learn preprocessing pipeline for the SPARCS dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.data.loader import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)

_PIPELINE_FILE = "preprocessing_pipeline.joblib"


def build_pipeline() -> ColumnTransformer:
    """Build a ColumnTransformer that handles numeric + categorical features."""
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) keeping only the expected feature columns."""
    # Ensure all expected feature columns exist
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning("Columns missing from DataFrame, will be imputed: %s", missing)
        for c in missing:
            df[c] = np.nan

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def save_pipeline(pipeline: ColumnTransformer, artifacts_dir: str | Path) -> Path:
    out = Path(artifacts_dir) / _PIPELINE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    logger.info("Preprocessing pipeline saved to %s", out)
    return out


def load_pipeline(artifacts_dir: str | Path) -> ColumnTransformer:
    path = Path(artifacts_dir) / _PIPELINE_FILE
    pipeline = joblib.load(path)
    logger.info("Preprocessing pipeline loaded from %s", path)
    return pipeline
