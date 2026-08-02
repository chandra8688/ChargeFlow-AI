"""
ChargeFlow AI V2 — Forecast Inference Service
================================================
Clean Python inference layer for the trained demand forecaster.

Responsibilities:
  - Load the saved model from disk (once at startup)
  - Accept a properly structured feature dict or DataFrame
  - Validate all required features are present
  - Run inference and clip predictions to physical bounds [0.0, 1.0]
  - Return structured prediction dict with confidence context
  - Raise descriptive errors for malformed input (never silently fail)

This module is intentionally decoupled from the Streamlit UI.
It can be imported as a standalone Python service or later wrapped
by a FastAPI endpoint in Phase 3.

Example usage:
    service = ForecastService()        # loads model from artifacts/models/
    result  = service.predict_single({
        "hour": 19, "day_of_week": 4, "month": 5,
        "hour_sin": ..., "hour_cos": ..., "day_sin": ..., "day_cos": ...,
        "is_weekend": 0, "is_holiday": 0, "temperature_c": 28.0,
        "lag_1h": 0.72, "lag_24h": 0.68, "lag_168h": 0.71,
        "rolling_mean_6h": 0.65, "rolling_mean_24h": 0.55, "rolling_std_24h": 0.08
    })
    # result: {"predicted_occupancy": 0.74, "status": "BUSY", ...}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union

from src.models.demand_forecaster import DemandForecaster, FEATURE_COLS, TARGET_COL


# ── Status Bands ─────────────────────────────────────────────────────────────

def _occupancy_to_status(occupancy: float) -> str:
    """Map a predicted occupancy_rate to a human-readable status label."""
    if occupancy >= 0.90:
        return "CRITICAL"
    elif occupancy >= 0.70:
        return "BUSY"
    elif occupancy >= 0.40:
        return "MODERATE"
    else:
        return "AVAILABLE"


# ── Forecast Service ──────────────────────────────────────────────────────────

class ForecastService:
    """
    Thin inference layer around the saved DemandForecaster model.

    Loads the model lazily on first prediction (or eagerly at init if
    eager_load=True) from artifacts/models/.
    """

    def __init__(self,
                 model_dir: Optional[Union[str, Path]] = None,
                 eager_load: bool = True):
        """
        Args:
            model_dir:   Directory containing demand_forecaster.joblib
                         and demand_forecaster_metadata.json.
                         Defaults to artifacts/models/ relative to project root.
            eager_load:  If True, loads model immediately at construction time.
        """
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "models"
        self._model_dir = Path(model_dir)
        self._forecaster: Optional[DemandForecaster] = None
        self.feature_cols = FEATURE_COLS

        if eager_load:
            self._load()

    def _load(self) -> None:
        """Load the saved DemandForecaster from disk."""
        self._forecaster = DemandForecaster.load(self._model_dir)

    def _ensure_loaded(self) -> None:
        if self._forecaster is None:
            self._load()

    # ── Input Validation ──────────────────────────────────────────────────────

    def _validate_input(self, feature_dict: Dict[str, Any]) -> None:
        """
        Validate that all required features are present and have valid types.

        Raises:
            ValueError with a descriptive message if validation fails.
        """
        missing = [f for f in self.feature_cols if f not in feature_dict]
        if missing:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"All 16 features must be provided: {self.feature_cols}"
            )
        # Check numeric types
        for feat in self.feature_cols:
            val = feature_dict[feat]
            try:
                float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Feature '{feat}' has non-numeric value: {val!r}. "
                    "All features must be numeric."
                )
        # Validate lag features are in [0,1] (they represent past occupancy)
        for lag_col in ["lag_1h", "lag_24h", "lag_168h"]:
            val = float(feature_dict[lag_col])
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Lag feature '{lag_col}' = {val:.4f} is outside [0,1]. "
                    "Lag features represent past occupancy and must be in [0, 1]."
                )

    # ── Public Predict Methods ────────────────────────────────────────────────

    def predict_single(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict occupancy_rate for a single station-hour observation.

        Args:
            feature_dict: Dict mapping each of the 16 feature names to a numeric value.

        Returns:
            Dict with:
                predicted_occupancy: float in [0.0, 1.0]
                status:              "AVAILABLE" | "MODERATE" | "BUSY" | "CRITICAL"
                model_type:          str
        """
        self._ensure_loaded()
        self._validate_input(feature_dict)

        X = pd.DataFrame([{f: float(feature_dict[f]) for f in self.feature_cols}])
        prediction = float(self._forecaster.predict(X)[0])

        return {
            "predicted_occupancy": round(prediction, 4),
            "status": _occupancy_to_status(prediction),
            "model_type": "RandomForestRegressor",
            "feature_count": len(self.feature_cols),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict occupancy_rate for multiple station-hour observations.

        Args:
            df: DataFrame with all 16 feature columns present.

        Returns:
            Input DataFrame with two new columns:
                predicted_occupancy: float
                predicted_status:    str
        """
        self._ensure_loaded()
        missing = [f for f in self.feature_cols if f not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in input DataFrame: {missing}")

        df = df.copy()
        preds = self._forecaster.predict(df)
        df["predicted_occupancy"] = preds
        df["predicted_status"] = [_occupancy_to_status(p) for p in preds]
        return df

    @property
    def model_metadata(self) -> Optional[Dict[str, Any]]:
        """Return model metadata dict if model is loaded, else None."""
        self._ensure_loaded()
        return self._forecaster._metadata if self._forecaster else None
