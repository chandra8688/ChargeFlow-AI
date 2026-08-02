"""
ChargeFlow AI V2 — Phase 2 Model Tests
========================================
Tests:
  1.  Model training executes without error
  2.  Expected feature columns are used (no leakage columns)
  3.  Target leakage is absent from feature set
  4.  Model can be saved to disk
  5.  Model can be reloaded from disk without retraining
  6.  Reloaded model produces identical predictions
  7.  Predictions lie within occupancy bounds [0.0, 1.0]
  8.  MAE, RMSE, R² compute correctly on a known toy dataset
  9.  SeasonalBaseline prediction equals lag_24h (by definition)
 10.  ForecastService validates and rejects malformed input
 11.  ForecastService predict_single returns correct structure
 12.  ForecastService predict_batch adds expected columns
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_timeseries import generate_hourly_timeseries
from src.data.preprocessor import DataPreprocessor
from src.models.demand_forecaster import (
    DemandForecaster, SeasonalBaseline, FEATURE_COLS, TARGET_COL, compute_metrics
)
from src.services.forecast_service import ForecastService


# ── Shared Fixture (module-level, generated once for speed) ───────────────────

def _build_fixtures():
    """Generate a minimal 21-day dataset and prepare splits for all tests."""
    df_raw = generate_hourly_timeseries(num_days=21, seed=99)
    prep = DataPreprocessor()
    df_feat = prep.engineer_features(df_raw)
    train_df, val_df, test_df = prep.split_chronological(df_feat)
    return df_raw, df_feat, train_df, val_df, test_df


_DF_RAW, _DF_FEAT, _TRAIN, _VAL, _TEST = _build_fixtures()


# ── Helper: minimal trained forecaster in a temp directory ────────────────────

def _train_mini_forecaster(tmp_dir: Path) -> DemandForecaster:
    fc = DemandForecaster(n_estimators=10, max_depth=4,
                          min_samples_leaf=5, random_state=0)
    fc.train(_TRAIN, _VAL)
    fc.save(model_dir=tmp_dir, train_df=_TRAIN, val_df=_VAL, test_df=_TEST)
    return fc


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestDemandForecasterCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = Path(tempfile.mkdtemp())
        cls.forecaster = _train_mini_forecaster(cls._tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    # 1. Model trains without error
    def test_01_training_executes(self):
        self.assertIsNotNone(self.forecaster.model)

    # 2. Expected feature columns are used
    def test_02_feature_columns_match_spec(self):
        self.assertEqual(self.forecaster.feature_cols, FEATURE_COLS)
        self.assertEqual(len(FEATURE_COLS), 16)

    # 3. No leakage columns in feature list
    def test_03_no_leakage_in_features(self):
        forbidden = {"occupancy_rate", "occupied_slots", "grid_load_kw"}
        for col in self.forecaster.feature_cols:
            self.assertNotIn(col, forbidden,
                msg=f"Leakage column '{col}' found in feature list.")

    # 4. Model can be saved
    def test_04_model_saved_to_disk(self):
        model_path = self._tmp / "demand_forecaster.joblib"
        meta_path  = self._tmp / "demand_forecaster_metadata.json"
        self.assertTrue(model_path.exists(), f"Model file not found: {model_path}")
        self.assertTrue(meta_path.exists(), f"Metadata file not found: {meta_path}")

    # 5. Model can be reloaded
    def test_05_model_loads_from_disk(self):
        loaded = DemandForecaster.load(self._tmp)
        self.assertIsNotNone(loaded.model)
        self.assertEqual(loaded.feature_cols, FEATURE_COLS)
        self.assertEqual(loaded.target_col, TARGET_COL)

    # 6. Reloaded model produces identical predictions
    def test_06_loaded_model_identical_predictions(self):
        X = _TEST[FEATURE_COLS].head(20)
        preds_orig   = self.forecaster.predict(X)
        loaded       = DemandForecaster.load(self._tmp)
        preds_loaded = loaded.predict(X)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)

    # 7. Predictions within [0, 1]
    def test_07_predictions_within_bounds(self):
        preds = self.forecaster.predict(_TEST[FEATURE_COLS].head(100))
        self.assertTrue((preds >= 0.0).all(), "Predictions below 0.0 found!")
        self.assertTrue((preds <= 1.0).all(), "Predictions above 1.0 found!")

    # 8. Metrics compute correctly on a known toy dataset
    def test_08_metrics_compute_correctly(self):
        y_true = np.array([0.5, 0.6, 0.7])
        y_pred = np.array([0.5, 0.6, 0.7])          # perfect predictions
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["MAE"], 0.0, places=5)
        self.assertAlmostEqual(m["RMSE"], 0.0, places=5)
        self.assertAlmostEqual(m["R2"], 1.0, places=4)

        y_pred_bad = np.array([1.0, 0.0, 1.0])       # bad predictions
        m_bad = compute_metrics(y_true, y_pred_bad)
        self.assertGreater(m_bad["MAE"], 0.0)
        self.assertLess(m_bad["R2"], 0.5)

    # 9. Metadata JSON is valid and complete
    def test_09_metadata_is_complete(self):
        meta_path = self._tmp / "demand_forecaster_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        for key in ["feature_cols", "target_col", "n_estimators",
                    "train_metrics", "val_metrics"]:
            self.assertIn(key, meta, f"Metadata missing key: {key}")
        self.assertEqual(meta["target_col"], TARGET_COL)
        self.assertEqual(meta["feature_cols"], FEATURE_COLS)


class TestSeasonalBaseline(unittest.TestCase):

    # 9. Baseline prediction = lag_24h by definition
    def test_baseline_prediction_equals_lag_24h(self):
        baseline = SeasonalBaseline()
        preds = baseline.predict(_TEST)
        expected = np.clip(_TEST["lag_24h"].values, 0.0, 1.0)
        np.testing.assert_allclose(preds, expected, rtol=1e-6)

    def test_baseline_metrics_return_dict(self):
        baseline = SeasonalBaseline()
        m = baseline.evaluate(_TEST)
        self.assertIn("MAE",  m)
        self.assertIn("RMSE", m)
        self.assertIn("R2",   m)
        self.assertGreater(m["MAE"], 0.0)


class TestForecastService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Use a minimal model trained in a temp dir for service tests."""
        cls._tmp = Path(tempfile.mkdtemp())
        _train_mini_forecaster(cls._tmp)
        cls.service = ForecastService(model_dir=cls._tmp, eager_load=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _valid_input(self):
        """Build a valid single-row feature dict from the test set."""
        row = _TEST[FEATURE_COLS].iloc[0].to_dict()
        return {k: float(v) for k, v in row.items()}

    # 10. Service rejects missing features
    def test_service_rejects_missing_feature(self):
        inp = self._valid_input()
        del inp["lag_1h"]
        with self.assertRaises(ValueError):
            self.service.predict_single(inp)

    # 10b. Service rejects invalid lag values
    def test_service_rejects_invalid_lag_value(self):
        inp = self._valid_input()
        inp["lag_1h"] = 1.5          # lag must be in [0,1]
        with self.assertRaises(ValueError):
            self.service.predict_single(inp)

    # 11. predict_single returns correct structure
    def test_service_predict_single_structure(self):
        result = self.service.predict_single(self._valid_input())
        self.assertIn("predicted_occupancy", result)
        self.assertIn("status", result)
        self.assertIn("model_type", result)
        pred = result["predicted_occupancy"]
        self.assertGreaterEqual(pred, 0.0)
        self.assertLessEqual(pred, 1.0)
        self.assertIn(result["status"], ["AVAILABLE", "MODERATE", "BUSY", "CRITICAL"])

    # 12. predict_batch adds expected columns
    def test_service_predict_batch_columns(self):
        batch_df = _TEST.head(10).copy()
        result_df = self.service.predict_batch(batch_df)
        self.assertIn("predicted_occupancy", result_df.columns)
        self.assertIn("predicted_status",    result_df.columns)
        self.assertTrue((result_df["predicted_occupancy"] >= 0.0).all())
        self.assertTrue((result_df["predicted_occupancy"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
