"""
ChargeFlow AI V2 — Phase 5 Explainability Tests
================================================
Tests for:
  - ExplainabilityService (global importance, top-N context, tree dispersion)
  - InferenceLogger (JSONL logging, thread safety, record schema)
  - GET /model/feature-importance API endpoint
  - POST /predict/raw/explain API endpoint
  - Regression: existing /predict/raw contract unchanged
  - Regression: all Phase 1–4 tests still pass (verified by discover run)

All tests work against the ACTUAL saved RandomForest artifact.
No metrics, importances, or dispersion values are hard-coded or fabricated.
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.models.demand_forecaster import FEATURE_COLS
from src.services.explainability_service import ExplainabilityService
from src.services.feature_service import FeatureService
from src.services.forecast_service import ForecastService
from src.services.inference_logger import InferenceLogger

# ── Shared fixtures (loaded once per module) ──────────────────────────────────

GOOD_STATION = "STA001"
GOOD_TIME    = "2025-06-15 19:00:00"
GOOD_TEMP    = 28.0
GOOD_HOLIDAY = False

_forecast_svc:     ForecastService | None     = None
_feature_svc:      FeatureService | None      = None
_explain_svc:      ExplainabilityService | None = None
_client_ctx:       TestClient                 = TestClient(app)


def setUpModule():
    global _forecast_svc, _feature_svc, _explain_svc
    _forecast_svc = ForecastService(eager_load=True)
    _feature_svc  = FeatureService()
    _explain_svc  = ExplainabilityService(_forecast_svc)
    _client_ctx.__enter__()


def tearDownModule():
    _client_ctx.__exit__(None, None, None)


def get_fc()   -> ForecastService:       return _forecast_svc
def get_fs()   -> FeatureService:        return _feature_svc
def get_es()   -> ExplainabilityService: return _explain_svc
def get_cli()  -> TestClient:            return _client_ctx


def _good_features():
    return get_fs().build_features(GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY)


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Global Feature Importance
# ══════════════════════════════════════════════════════════════════════════════

class TestGlobalFeatureImportance(unittest.TestCase):

    def test_01_returns_exactly_16_entries(self):
        """global_feature_importance() must return exactly 16 items."""
        result = get_es().global_feature_importance()
        self.assertEqual(len(result), 16,
            f"Expected 16 importances, got {len(result)}")

    def test_02_feature_names_match_feature_cols(self):
        """Names in importance list must match FEATURE_COLS exactly (any order)."""
        result  = get_es().global_feature_importance()
        names   = [item["feature"] for item in result]
        self.assertEqual(sorted(names), sorted(FEATURE_COLS),
            f"Feature name mismatch.\nExpected: {sorted(FEATURE_COLS)}\nGot:      {sorted(names)}")

    def test_03_importances_in_0_1(self):
        """All MDI importance values must be in [0.0, 1.0]."""
        result = get_es().global_feature_importance()
        for item in result:
            self.assertGreaterEqual(item["importance"], 0.0,
                f"Importance < 0 for feature {item['feature']}")
            self.assertLessEqual(item["importance"], 1.0,
                f"Importance > 1 for feature {item['feature']}")

    def test_04_importances_sum_approx_one(self):
        """MDI importances must sum to approximately 1.0 (float tolerance)."""
        result = get_es().global_feature_importance()
        total  = sum(item["importance"] for item in result)
        self.assertAlmostEqual(total, 1.0, places=3,
            msg=f"Importances sum = {total:.5f}, expected ≈ 1.0")

    def test_05_sorted_descending(self):
        """Importance list must be sorted in descending order."""
        result = get_es().global_feature_importance()
        importances = [item["importance"] for item in result]
        self.assertEqual(importances, sorted(importances, reverse=True),
            "Importances are not sorted in descending order")

    def test_06_result_is_cached(self):
        """Calling global_feature_importance() twice returns the same object."""
        r1 = get_es().global_feature_importance()
        r2 = get_es().global_feature_importance()
        self.assertIs(r1, r2, "Cache did not return the same object on second call")

    def test_07_hour_is_top_feature(self):
        """
        'hour' is known to be the dominant MDI feature in this dataset.
        This test verifies the model is the actual saved artifact — not a stub.
        """
        result    = get_es().global_feature_importance()
        top_name  = result[0]["feature"]
        self.assertEqual(top_name, "hour",
            f"Expected top feature 'hour', got '{top_name}'")
        self.assertGreater(result[0]["importance"], 0.5,
            "Top feature 'hour' has suspiciously low importance for this dataset")


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Top-N Feature Context
# ══════════════════════════════════════════════════════════════════════════════

class TestTopNFeatureContext(unittest.TestCase):

    def test_08_top_5_returns_exactly_5(self):
        feat = _good_features()
        ctx  = get_es().top_n_feature_context(feat, n=5)
        self.assertEqual(len(ctx), 5)

    def test_09_each_item_has_required_keys(self):
        feat = _good_features()
        ctx  = get_es().top_n_feature_context(feat, n=5)
        for item in ctx:
            self.assertIn("feature", item)
            self.assertIn("importance", item)
            self.assertIn("value", item)

    def test_10_values_match_feature_dict(self):
        """value in each context item must equal the input in feature_dict."""
        feat = _good_features()
        ctx  = get_es().top_n_feature_context(feat, n=5)
        for item in ctx:
            expected = round(float(feat[item["feature"]]), 6)
            self.assertAlmostEqual(item["value"], expected, places=5,
                msg=f"Context value mismatch for feature '{item['feature']}'")

    def test_11_context_features_are_top_by_mdi(self):
        """Context features must be the top-5 from global_feature_importance()."""
        importances = get_es().global_feature_importance()
        expected_top5 = [item["feature"] for item in importances[:5]]
        feat = _good_features()
        ctx  = get_es().top_n_feature_context(feat, n=5)
        actual_names = [item["feature"] for item in ctx]
        self.assertEqual(actual_names, expected_top5,
            f"Context features do not match top-5 MDI features.\n"
            f"Expected: {expected_top5}\nGot:      {actual_names}")


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — Tree Prediction Spread
# ══════════════════════════════════════════════════════════════════════════════

class TestTreeDispersion(unittest.TestCase):

    def _disp(self):
        return get_es().tree_dispersion(_good_features())

    def test_12_required_keys_present(self):
        """tree_dispersion() must return all required keys."""
        d = self._disp()
        required = {"tree_mean", "tree_std", "tree_min", "tree_max",
                    "p10", "p90", "estimator_count", "status_consensus_pct",
                    "disclaimer"}
        missing = required - set(d.keys())
        self.assertFalse(missing, f"Missing keys: {missing}")

    def test_13_estimator_count_matches_rf_model(self):
        """estimator_count must equal the number of trees in the saved RF model."""
        d = self._disp()
        rf = get_fc()._forecaster.model
        self.assertEqual(d["estimator_count"], len(rf.estimators_))

    def test_14_tree_mean_consistent_with_predict_single(self):
        """
        tree_mean must be consistent with ForecastService.predict_single()
        to within float rounding.  Both are mean of 200 clipped tree predictions.
        """
        feat = _good_features()
        d    = get_es().tree_dispersion(feat)
        pred = get_fc().predict_single(feat)["predicted_occupancy"]
        self.assertAlmostEqual(d["tree_mean"], pred, places=3,
            msg=f"tree_mean {d['tree_mean']:.4f} ≠ predict_single {pred:.4f}")

    def test_15_ordering_invariants(self):
        """tree_min ≤ p10 ≤ tree_mean ≤ p90 ≤ tree_max."""
        d = self._disp()
        self.assertLessEqual(d["tree_min"],  d["p10"],
            "tree_min > p10")
        self.assertLessEqual(d["p10"],       d["tree_mean"],
            "p10 > tree_mean — possible only with extreme skew; verify input")
        self.assertLessEqual(d["tree_mean"], d["p90"],
            "tree_mean > p90")
        self.assertLessEqual(d["p90"],       d["tree_max"],
            "p90 > tree_max")

    def test_15b_min_max_within_unit_interval(self):
        d = self._disp()
        self.assertGreaterEqual(d["tree_min"], 0.0)
        self.assertLessEqual(d["tree_max"],    1.0)

    def test_16_consensus_bounded(self):
        """status_consensus_pct must be in [0.0, 100.0]."""
        d = self._disp()
        self.assertGreaterEqual(d["status_consensus_pct"],  0.0)
        self.assertLessEqual(d["status_consensus_pct"],   100.0)

    def test_17_std_non_negative(self):
        d = self._disp()
        self.assertGreaterEqual(d["tree_std"], 0.0)

    def test_18_disclaimer_is_present_and_non_empty(self):
        d = self._disp()
        self.assertIn("disclaimer", d)
        self.assertGreater(len(d["disclaimer"]), 20,
            "Disclaimer string is suspiciously short")

    def test_19_evening_peak_high_consensus(self):
        """
        STA001 at 19:00 is known to be a high-occupancy peak.
        Expect >90% of trees to agree on the same status bucket.
        """
        d = self._disp()
        self.assertGreater(d["status_consensus_pct"], 90.0,
            f"Expected >90% consensus at peak hour; got {d['status_consensus_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — InferenceLogger
# ══════════════════════════════════════════════════════════════════════════════

class TestInferenceLogger(unittest.TestCase):

    def _make_logger(self):
        """Create a temporary InferenceLogger for test isolation."""
        tmp = tempfile.mkdtemp()
        return InferenceLogger(log_path=Path(tmp) / "test_inference.jsonl")

    def _log_one(self, logger: InferenceLogger) -> str:
        return logger.log(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            predicted_occupancy=0.985,
            status="CRITICAL",
            model_version="2026-08-01T20:45:13.289266Z",
            inference_latency_ms=14.7,
            source="test",
            key_features={
                "hour": 19, "lag_1h": 1.0, "lag_24h": 1.0,
                "lag_168h": 1.0, "rolling_mean_24h": 0.604,
                "temperature_c": 28.0,
            },
        )

    def test_20_log_creates_file(self):
        logger = self._make_logger()
        self.assertFalse(logger.log_path().exists())
        self._log_one(logger)
        self.assertTrue(logger.log_path().exists())

    def test_21_log_record_valid_json(self):
        logger = self._make_logger()
        self._log_one(logger)
        lines = logger.log_path().read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertIsInstance(record, dict)

    def test_22_required_fields_present(self):
        logger = self._make_logger()
        self._log_one(logger)
        record = json.loads(
            logger.log_path().read_text(encoding="utf-8").strip().splitlines()[0]
        )
        required = {
            "prediction_id", "logged_at", "station_id", "prediction_time",
            "predicted_occupancy", "status", "model_version",
            "inference_latency_ms", "source", "key_features",
        }
        missing = required - set(record.keys())
        self.assertFalse(missing, f"Missing log fields: {missing}")

    def test_23_model_version_matches_artifact(self):
        """model_version in log record must match the saved artifact's trained_at."""
        meta = get_fc().model_metadata or {}
        trained_at = meta.get("trained_at", "")
        self.assertTrue(len(trained_at) > 0,
            "trained_at not found in model metadata artifact")

        logger = self._make_logger()
        logger.log(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            predicted_occupancy=0.985,
            status="CRITICAL",
            model_version=trained_at,
            inference_latency_ms=10.0,
            source="test",
            key_features={},
        )
        record = json.loads(
            logger.log_path().read_text(encoding="utf-8").strip().splitlines()[0]
        )
        self.assertEqual(record["model_version"], trained_at)

    def test_24_prediction_id_is_unique(self):
        """Each log() call must return a unique prediction_id."""
        logger = self._make_logger()
        ids = {self._log_one(logger) for _ in range(10)}
        self.assertEqual(len(ids), 10, "Duplicate prediction_id generated")

    def test_25_thread_safety(self):
        """Concurrent log() calls from multiple threads must produce valid records."""
        logger  = self._make_logger()
        n_threads = 20
        errors  = []

        def _write():
            try:
                self._log_one(logger)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertFalse(errors, f"Errors during concurrent writes: {errors}")
        df = logger.read_log()
        self.assertEqual(len(df), n_threads,
            f"Expected {n_threads} records, got {len(df)}")

    def test_26_read_log_returns_dataframe(self):
        import pandas as pd
        logger = self._make_logger()
        empty  = logger.read_log()
        self.assertIsInstance(empty, pd.DataFrame)
        self.assertEqual(len(empty), 0)

        self._log_one(logger)
        df = logger.read_log()
        self.assertEqual(len(df), 1)
        self.assertIn("prediction_id", df.columns)


# ══════════════════════════════════════════════════════════════════════════════
# API Tests — GET /model/feature-importance
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureImportanceEndpoint(unittest.TestCase):

    def test_27_returns_200(self):
        resp = get_cli().get("/model/feature-importance")
        self.assertEqual(resp.status_code, 200,
            f"Expected 200, got {resp.status_code}. Body: {resp.text}")

    def test_28_n_features_is_16(self):
        resp = get_cli().get("/model/feature-importance")
        body = resp.json()
        self.assertEqual(body["n_features"], 16)
        self.assertEqual(len(body["importances"]), 16)

    def test_29_feature_names_match_feature_cols(self):
        resp   = get_cli().get("/model/feature-importance")
        body   = resp.json()
        names  = [item["feature"] for item in body["importances"]]
        self.assertEqual(sorted(names), sorted(FEATURE_COLS))

    def test_30_importances_valid(self):
        resp = get_cli().get("/model/feature-importance")
        body = resp.json()
        for item in body["importances"]:
            self.assertGreaterEqual(item["importance"], 0.0)
            self.assertLessEqual(item["importance"],   1.0)

    def test_31_importances_sorted_descending(self):
        resp   = get_cli().get("/model/feature-importance")
        body   = resp.json()
        vals   = [item["importance"] for item in body["importances"]]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_32_model_type_present(self):
        resp = get_cli().get("/model/feature-importance")
        body = resp.json()
        self.assertIn("model_type", body)
        self.assertGreater(len(body["model_type"]), 0)


# ══════════════════════════════════════════════════════════════════════════════
# API Tests — POST /predict/raw/explain
# ══════════════════════════════════════════════════════════════════════════════

_EXPLAIN_PAYLOAD = {
    "station_id":      GOOD_STATION,
    "prediction_time": GOOD_TIME,
    "temperature_c":   GOOD_TEMP,
    "is_holiday":      GOOD_HOLIDAY,
}


class TestRawExplainEndpoint(unittest.TestCase):

    def test_33_returns_200(self):
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        self.assertEqual(resp.status_code, 200,
            f"Expected 200, got {resp.status_code}. Body: {resp.text}")

    def test_34_response_schema_complete(self):
        """All expected top-level fields must be present."""
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        body = resp.json()
        for key in ("prediction_id", "station_id", "prediction_time",
                    "predicted_occupancy", "status", "model_type",
                    "feature_context", "top_feature_context",
                    "tree_dispersion", "inference_latency_ms"):
            self.assertIn(key, body, f"Missing field: '{key}'")

    def test_35_tree_dispersion_fields(self):
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        td   = resp.json()["tree_dispersion"]
        for key in ("tree_mean", "tree_std", "tree_min", "tree_max",
                    "p10", "p90", "estimator_count", "status_consensus_pct",
                    "disclaimer"):
            self.assertIn(key, td, f"Missing tree_dispersion field: '{key}'")

    def test_36_top_feature_context_has_5_items(self):
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        body = resp.json()
        self.assertEqual(len(body["top_feature_context"]), 5)

    def test_37_predicted_occupancy_matches_predict_raw(self):
        """
        /predict/raw/explain must return the same predicted_occupancy
        as /predict/raw for the same input — the model contract is unchanged.
        """
        r_raw     = get_cli().post("/predict/raw",         json=_EXPLAIN_PAYLOAD)
        r_explain = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        self.assertEqual(r_raw.status_code,     200)
        self.assertEqual(r_explain.status_code, 200)
        occ_raw     = r_raw.json()["predicted_occupancy"]
        occ_explain = r_explain.json()["predicted_occupancy"]
        self.assertAlmostEqual(occ_raw, occ_explain, places=4,
            msg="/predict/raw and /predict/raw/explain returned different occupancy values")

    def test_38_tree_mean_consistent_with_predicted_occupancy(self):
        """tree_mean must match predicted_occupancy within float rounding."""
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        body = resp.json()
        self.assertAlmostEqual(
            body["tree_dispersion"]["tree_mean"],
            body["predicted_occupancy"],
            places=3,
            msg="tree_mean is inconsistent with predicted_occupancy",
        )

    def test_39_estimator_count_correct(self):
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        ec   = resp.json()["tree_dispersion"]["estimator_count"]
        rf   = get_fc()._forecaster.model
        self.assertEqual(ec, len(rf.estimators_))

    def test_40_prediction_id_is_uuid(self):
        import re
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        pid  = resp.json()["prediction_id"]
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        self.assertTrue(uuid_re.match(pid), f"prediction_id is not a valid uuid4: {pid}")

    def test_41_latency_is_positive(self):
        resp = get_cli().post("/predict/raw/explain", json=_EXPLAIN_PAYLOAD)
        ms   = resp.json()["inference_latency_ms"]
        self.assertGreater(ms, 0.0)

    def test_42_unknown_station_returns_404(self):
        payload = dict(_EXPLAIN_PAYLOAD, station_id="STA999")
        resp    = get_cli().post("/predict/raw/explain", json=payload)
        self.assertEqual(resp.status_code, 404)

    def test_43_insufficient_history_returns_422(self):
        payload = dict(_EXPLAIN_PAYLOAD, prediction_time="2025-01-02 00:00:00")
        resp    = get_cli().post("/predict/raw/explain", json=payload)
        self.assertEqual(resp.status_code, 422)


# ══════════════════════════════════════════════════════════════════════════════
# Regression — Existing /predict/raw Contract Unchanged
# ══════════════════════════════════════════════════════════════════════════════

class TestRawPredictRegressionPhase5(unittest.TestCase):

    def test_44_predict_raw_still_returns_200(self):
        resp = get_cli().post("/predict/raw", json=_EXPLAIN_PAYLOAD)
        self.assertEqual(resp.status_code, 200)

    def test_45_predict_raw_schema_unchanged(self):
        resp = get_cli().post("/predict/raw", json=_EXPLAIN_PAYLOAD)
        body = resp.json()
        for key in ("station_id", "prediction_time",
                    "predicted_occupancy", "status",
                    "model_type", "feature_context"):
            self.assertIn(key, body)
        # Phase 5 fields must NOT appear in /predict/raw response
        self.assertNotIn("prediction_id",    body)
        self.assertNotIn("top_feature_context", body)
        self.assertNotIn("tree_dispersion",  body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
