"""SHAP-based cost explainer for tree and linear models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CostExplainer:
    """Wraps shap.TreeExplainer / LinearExplainer for the trained cost model."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self._explainer: Any = None
        self._kind = "none"

        try:
            import shap

            # Covers XGBRegressor, LGBMRegressor, RandomForestRegressor
            tree_attrs = ("get_booster", "booster_", "estimators_")
            if any(hasattr(model, a) for a in tree_attrs):
                self._explainer = shap.TreeExplainer(model)
                self._kind = "tree"
                logger.info("SHAP TreeExplainer initialised.")
            elif hasattr(model, "coef_"):            # Ridge / linear
                self._explainer = shap.LinearExplainer(
                    model, np.zeros((1, len(feature_names)))
                )
                self._kind = "linear"
                logger.info("SHAP LinearExplainer initialised.")
            else:
                logger.warning("No SHAP explainer available for %s.", type(model).__name__)
        except Exception as exc:
            logger.warning("SHAP initialisation failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._explainer is not None

    # ── Core methods ──────────────────────────────────────────────────────────

    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        """Return raw SHAP values array of shape (n_samples, n_features)."""
        if not self.available:
            return None
        try:
            vals = self._explainer.shap_values(X, check_additivity=False)
            if isinstance(vals, list):
                vals = vals[0]
            return vals
        except Exception as exc:
            logger.warning("shap_values() failed: %s", exc)
            return None

    def expected_value(self) -> float:
        if not self.available:
            return 0.0
        ev = self._explainer.expected_value
        return float(ev[0]) if hasattr(ev, "__len__") else float(ev)

    # ── Convenience helpers ───────────────────────────────────────────────────

    def waterfall_data(self, X_row: np.ndarray) -> pd.DataFrame:
        """SHAP values for a single prediction row, sorted by absolute impact."""
        vals = self.shap_values(X_row)
        if vals is None:
            return pd.DataFrame()
        row = vals[0] if vals.ndim > 1 else vals
        df = pd.DataFrame(
            {"Feature": self.feature_names, "SHAP Value": row}
        )
        df["Direction"] = df["SHAP Value"].apply(
            lambda v: "Increases Cost" if v >= 0 else "Decreases Cost"
        )
        df["abs"] = df["SHAP Value"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").head(12).reset_index(drop=True)

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Mean absolute SHAP across rows — global feature importance."""
        vals = self.shap_values(X)
        if vals is None:
            return pd.DataFrame()
        mean_abs = np.abs(vals).mean(axis=0)
        return (
            pd.DataFrame({"Feature": self.feature_names, "Importance": mean_abs})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
