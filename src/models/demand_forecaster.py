"""
ChargeFlow AI V2 — Demand Forecaster
======================================
Implements:
  1. SeasonalBaseline  — Lag-24h naive seasonal persistence baseline
  2. DemandForecaster  — RandomForest regression model for hourly occupancy

Design principles:
  - Target: occupancy_rate = occupied_slots / total_slots  (0.0 – 1.0)
  - No future information or target leakage in features
  - Strict chronological train/validation/test discipline
  - Deterministic random seed for reproducibility
  - Native scikit-learn feature importance for explainability
  - joblib persistence (save/load without retraining)

Features used (16 total):
  hour, day_of_week, month,
  hour_sin, hour_cos, day_sin, day_cos,
  is_weekend, is_holiday, temperature_c,
  lag_1h, lag_24h, lag_168h,
  rolling_mean_6h, rolling_mean_24h, rolling_std_24h

Explicitly EXCLUDED from model input:
  - occupancy_rate (the target itself)
  - occupied_slots, grid_load_kw (directly derived from target)
  - station_id, timestamp, city, charger_type (metadata, not fed to RF directly)
  - total_slots (constant per station; captured via lag patterns)
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_weekend", "is_holiday", "temperature_c",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h",
]
TARGET_COL: str = "occupancy_rate"

# Columns that must NEVER appear as features (leakage guard)
_FORBIDDEN_FEATURES = {
    "occupancy_rate", "occupied_slots", "grid_load_kw",
}

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"


# ── Evaluation Utilities ──────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for a set of predictions."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 5), "RMSE": round(rmse, 5), "R2": round(r2, 4)}


# ── Seasonal Naive Baseline ───────────────────────────────────────────────────

class SeasonalBaseline:
    """
    Lag-24h Seasonal Persistence Baseline.

    Prediction rule: occupancy(t) = occupancy(t - 24h)

    This is the standard "same hour yesterday" baseline for hourly time-series.
    It is the minimum bar any real forecasting model must beat to be useful.
    It requires no training — it simply uses the lag_24h feature already in the
    dataset, making it fast and interpretable.
    """

    def __init__(self):
        self.name = "Seasonal Lag-24h Baseline"
        self._metrics: Optional[Dict[str, float]] = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns lag_24h values as predictions.

        Args:
            df: DataFrame containing a 'lag_24h' column (from Phase 1 preprocessor).

        Returns:
            numpy array of predictions, clipped to [0, 1].
        """
        if "lag_24h" not in df.columns:
            raise ValueError("'lag_24h' column is required for SeasonalBaseline.")
        return np.clip(df["lag_24h"].values, 0.0, 1.0)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate baseline on a split DataFrame containing both lag_24h and target."""
        y_true = df[TARGET_COL].values
        y_pred = self.predict(df)
        self._metrics = compute_metrics(y_true, y_pred)
        return self._metrics


# ── Random Forest Demand Forecaster ──────────────────────────────────────────

class DemandForecaster:
    """
    RandomForest Regressor for hourly EV charging demand forecasting.

    Training flow:
        1. Validate that FEATURE_COLS exist and no forbidden features are present.
        2. Fit RandomForestRegressor on training set (X_train, y_train).
        3. Evaluate on validation set for model selection.
        4. Evaluate on held-out test set for final reporting.

    Persistence:
        - Model binary saved to artifacts/models/demand_forecaster.joblib
        - Metadata (features, metrics, dates) saved to
          artifacts/models/demand_forecaster_metadata.json
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 16,
                 min_samples_leaf: int = 10, random_state: int = 42):
        """
        Args:
            n_estimators:     Number of trees in the forest.
            max_depth:        Max tree depth (controls overfitting).
            min_samples_leaf: Min samples required at leaf node.
            random_state:     Fixed seed for reproducibility.
        """
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state

        self.model: Optional[RandomForestRegressor] = None
        self.feature_cols: List[str] = FEATURE_COLS
        self.target_col: str = TARGET_COL
        self._train_metrics: Optional[Dict[str, float]] = None
        self._val_metrics:   Optional[Dict[str, float]] = None
        self._test_metrics:  Optional[Dict[str, float]] = None
        self._metadata: Dict[str, Any] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Guard against missing features or target leakage."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        leaked = [c for c in self.feature_cols if c in _FORBIDDEN_FEATURES]
        if leaked:
            raise ValueError(f"Forbidden leakage columns in feature list: {leaked}")

    def _get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix X and target series y from a split DataFrame."""
        self._validate_features(df)
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        return X, y

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the RandomForest model.

        Args:
            train_df: Training split (from DataPreprocessor.split_chronological).
            val_df:   Validation split (for model selection / hyperparameter reference).

        Returns:
            Dict containing training and validation metrics.
        """
        X_train, y_train = self._get_X_y(train_df)
        X_val,   y_val   = self._get_X_y(val_df)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

        print(f"Training RandomForestRegressor on {len(X_train):,} samples "
              f"({len(self.feature_cols)} features)...")
        self.model.fit(X_train, y_train)

        # Evaluate on training set
        y_train_pred = np.clip(self.model.predict(X_train), 0.0, 1.0)
        self._train_metrics = compute_metrics(y_train.values, y_train_pred)

        # Evaluate on validation set
        y_val_pred = np.clip(self.model.predict(X_val), 0.0, 1.0)
        self._val_metrics = compute_metrics(y_val.values, y_val_pred)

        print(f"  Train  — MAE: {self._train_metrics['MAE']:.4f}  "
              f"RMSE: {self._train_metrics['RMSE']:.4f}  "
              f"R²: {self._train_metrics['R2']:.4f}")
        print(f"  Val    — MAE: {self._val_metrics['MAE']:.4f}  "
              f"RMSE: {self._val_metrics['RMSE']:.4f}  "
              f"R²: {self._val_metrics['R2']:.4f}")

        self._metadata["trained_at"] = datetime.now(timezone.utc).isoformat()
        return {"train": self._train_metrics, "val": self._val_metrics}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_test(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on the held-out test set.

        IMPORTANT: Call this ONLY after training is complete and hyperparameters
        are finalised. The test set must never be used during model selection.

        Returns:
            Dict with MAE, RMSE, R².
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call .train() first.")
        X_test, y_test = self._get_X_y(test_df)
        y_pred = np.clip(self.model.predict(X_test), 0.0, 1.0)
        self._test_metrics = compute_metrics(y_test.values, y_pred)
        print(f"  Test   — MAE: {self._test_metrics['MAE']:.4f}  "
              f"RMSE: {self._test_metrics['RMSE']:.4f}  "
              f"R²: {self._test_metrics['R2']:.4f}")
        return self._test_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run inference on a pre-engineered feature DataFrame.

        Returns:
            numpy array of occupancy_rate predictions, clipped to [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call .train() or load() first.")
        self._validate_features(X)
        raw_pred = self.model.predict(X[self.feature_cols])
        return np.clip(raw_pred, 0.0, 1.0)

    # ── Error Analysis ────────────────────────────────────────────────────────

    def error_analysis(self, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Breakdown of prediction errors by city, hour, and charger_type.

        Args:
            test_df: Test split DataFrame (with metadata columns still present).

        Returns:
            Dict of DataFrames: {'by_city', 'by_hour', 'by_charger_type'}
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before error analysis.")

        df = test_df.copy()
        X, y_true = self._get_X_y(df)
        y_pred = np.clip(self.model.predict(X), 0.0, 1.0)
        df["_pred"] = y_pred
        df["_abs_error"] = np.abs(y_true.values - y_pred)

        results = {}

        # By City
        if "city" in df.columns:
            by_city = df.groupby("city").apply(
                lambda g: pd.Series(compute_metrics(
                    g[TARGET_COL].values, g["_pred"].values
                ))
            ).reset_index()
            results["by_city"] = by_city

        # By Hour
        if "hour" in df.columns:
            by_hour = df.groupby("hour").apply(
                lambda g: pd.Series(compute_metrics(
                    g[TARGET_COL].values, g["_pred"].values
                ))
            ).reset_index()
            results["by_hour"] = by_hour

        # By Charger Type
        if "charger_type" in df.columns:
            by_charger = df.groupby("charger_type").apply(
                lambda g: pd.Series(compute_metrics(
                    g[TARGET_COL].values, g["_pred"].values
                ))
            ).reset_index()
            results["by_charger_type"] = by_charger

        return results

    # ── Feature Importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """
        Return native RandomForest feature importances (Gini impurity reduction).

        NOTE: These are Mean Decrease in Impurity (MDI) values.
        Interpretation: Higher importance = stronger predictive association with
        occupancy_rate. These are NOT causal claims — they reflect predictive
        association learned from the training data.

        Returns:
            DataFrame with 'feature' and 'importance' columns, sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained first.")
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df["importance"] = fi_df["importance"].round(5)
        return fi_df

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Optional[Path] = None,
             train_df: Optional[pd.DataFrame] = None,
             val_df:   Optional[pd.DataFrame] = None,
             test_df:  Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Save the trained model binary and metadata JSON.

        Args:
            model_dir: Directory to save artifacts (default: artifacts/models/).
            train_df, val_df, test_df: Optional DataFrames to record date ranges.

        Returns:
            Dict with paths to saved model and metadata.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before saving.")

        if model_dir is None:
            model_dir = ARTIFACTS_DIR / "models"
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path    = model_dir / "demand_forecaster.joblib"
        metadata_path = model_dir / "demand_forecaster_metadata.json"

        # Save model binary
        joblib.dump(self.model, model_path)

        # Build metadata dict
        def ts_range(df):
            if df is not None and "timestamp" in df.columns:
                return {
                    "min": str(df["timestamp"].min()),
                    "max": str(df["timestamp"].max()),
                    "rows": len(df),
                }
            return {}

        metadata = {
            "model_type":      "RandomForestRegressor (scikit-learn)",
            "trained_at":      self._metadata.get("trained_at", ""),
            "random_state":    self.random_state,
            "n_estimators":    self.n_estimators,
            "max_depth":       self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_cols":    self.feature_cols,
            "target_col":      self.target_col,
            "n_features":      len(self.feature_cols),
            "train_metrics":   self._train_metrics or {},
            "val_metrics":     self._val_metrics   or {},
            "test_metrics":    self._test_metrics  or {},
            "train_dates":     ts_range(train_df),
            "val_dates":       ts_range(val_df),
            "test_dates":      ts_range(test_df),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved   : {model_path}")
        print(f"Metadata saved: {metadata_path}")

        return {"model_path": str(model_path), "metadata_path": str(metadata_path)}

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "DemandForecaster":
        """
        Load a previously saved model from disk.

        Args:
            model_dir: Directory containing the saved artifacts.

        Returns:
            DemandForecaster instance with model ready for inference.
        """
        if model_dir is None:
            model_dir = ARTIFACTS_DIR / "models"
        model_dir = Path(model_dir)

        model_path    = model_dir / "demand_forecaster.joblib"
        metadata_path = model_dir / "demand_forecaster_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        forecaster = cls(
            n_estimators=meta.get("n_estimators", 200),
            max_depth=meta.get("max_depth", 16),
            min_samples_leaf=meta.get("min_samples_leaf", 10),
            random_state=meta.get("random_state", 42),
        )
        forecaster.model          = joblib.load(model_path)
        forecaster.feature_cols   = meta.get("feature_cols", FEATURE_COLS)
        forecaster.target_col     = meta.get("target_col", TARGET_COL)
        forecaster._train_metrics = meta.get("train_metrics")
        forecaster._val_metrics   = meta.get("val_metrics")
        forecaster._test_metrics  = meta.get("test_metrics")
        forecaster._metadata      = meta

        return forecaster
