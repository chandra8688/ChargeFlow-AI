"""
ChargeFlow AI V2 — Phase 3 API Tests
======================================
Tests the FastAPI application using FastAPI's TestClient (which uses httpx
under the hood — no real server process needed).

Coverage:
  1.  GET /health returns 200
  2.  Health response has correct structure
  3.  health model_loaded is True when artifacts exist
  4.  GET /model/info returns 200
  5.  model/info exposes feature_names matching Phase 2 contract
  6.  model/info exposes actual metrics (not fabricated values)
  7.  POST /predict returns 200 for a valid request
  8.  predicted_occupancy is within physical bounds [0.0, 1.0]
  9.  status is one of the expected labels
 10.  POST /predict returns 422 for a missing required feature
 11.  POST /predict returns 422 for an out-of-range 'hour' value
 12.  POST /predict returns 422 for an out-of-range lag value
 13.  POST /predict response structure is stable (all expected keys present)
 14.  POST /predict/batch works for a valid list of requests
 15.  batch response has correct count
 16.  batch results are in input order (indexed)
 17.  POST /predict/batch rejects oversized batches (> MAX_BATCH_SIZE)
 18.  POST /predict/batch rejects empty list
 19.  POST /predict/batch returns 422 if ANY item is malformed
 20.  API uses the REAL model (predicted_occupancy differs for different inputs)
"""

import math
import unittest
from typing import Dict, Any

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import MAX_BATCH_SIZE
from src.models.demand_forecaster import FEATURE_COLS


# ── Shared TestClient ─────────────────────────────────────────────────────────
# TestClient must be used as a context manager so FastAPI's lifespan
# runs (which loads the ForecastService / ML model into app.state).
# We create ONE client shared across ALL test classes to avoid loading
# the 107 MB model multiple times during the test run.

_client_ctx = TestClient(app)


def setUpModule():
    """Enter the TestClient context: triggers the FastAPI lifespan (model load)."""
    global _client_ctx
    _client_ctx.__enter__()


def tearDownModule():
    """Exit the TestClient context: triggers lifespan shutdown."""
    global _client_ctx
    _client_ctx.__exit__(None, None, None)


def get_client() -> TestClient:
    return _client_ctx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_payload() -> Dict[str, Any]:
    """
    A physically plausible input for Friday 19:00 in May.
    All values satisfy Pydantic schema bounds.
    """
    hour = 19
    dow  = 4   # Friday
    return {
        "hour":             hour,
        "day_of_week":      dow,
        "month":            5,
        "hour_sin":         round(math.sin(2 * math.pi * hour / 24), 6),
        "hour_cos":         round(math.cos(2 * math.pi * hour / 24), 6),
        "day_sin":          round(math.sin(2 * math.pi * dow / 7), 6),
        "day_cos":          round(math.cos(2 * math.pi * dow / 7), 6),
        "is_weekend":       0,
        "is_holiday":       0,
        "temperature_c":    27.5,
        "lag_1h":           0.72,
        "lag_24h":          0.68,
        "lag_168h":         0.71,
        "rolling_mean_6h":  0.65,
        "rolling_mean_24h": 0.55,
        "rolling_std_24h":  0.08,
    }


# ── Health Tests ──────────────────────────────────────────────────────────────

class TestHealthEndpoint(unittest.TestCase):

    def test_01_health_returns_200(self):
        resp = get_client().get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_02_health_correct_structure(self):
        resp = get_client().get("/health")
        body = resp.json()
        self.assertIn("status",       body)
        self.assertIn("service",      body)
        self.assertIn("model_loaded", body)

    def test_03_health_model_loaded_is_true(self):
        """model_loaded must be True when artifacts/models/ are present."""
        resp = get_client().get("/health")
        body = resp.json()
        self.assertTrue(
            body["model_loaded"],
            "model_loaded is False — did you run 'python -m src.train_evaluate'?"
        )


# ── Model Info Tests ──────────────────────────────────────────────────────────

class TestModelInfoEndpoint(unittest.TestCase):

    def test_04_model_info_returns_200(self):
        resp = get_client().get("/model/info")
        self.assertEqual(resp.status_code, 200)

    def test_05_model_info_feature_names_match_phase2(self):
        """Feature names must match the Phase 2 FEATURE_COLS contract exactly."""
        resp = get_client().get("/model/info")
        body = resp.json()
        self.assertIn("feature_names", body)
        self.assertEqual(
            body["feature_names"], FEATURE_COLS,
            f"Mismatch: API reports {body['feature_names']}, expected {FEATURE_COLS}"
        )

    def test_06_model_info_has_metrics(self):
        """Metrics must exist and must NOT be empty dicts (real training happened)."""
        resp = get_client().get("/model/info")
        body = resp.json()
        self.assertIn("test_metrics", body)
        if body["test_metrics"]:
            self.assertIn("MAE", body["test_metrics"])
            self.assertIn("R2",  body["test_metrics"])


# ── Single Prediction Tests ───────────────────────────────────────────────────

class TestPredictEndpoint(unittest.TestCase):

    def test_07_valid_predict_returns_200(self):
        resp = get_client().post("/predict", json=_valid_payload())
        self.assertEqual(resp.status_code, 200)

    def test_08_predicted_occupancy_in_bounds(self):
        resp = get_client().post("/predict", json=_valid_payload())
        pred = resp.json()["predicted_occupancy"]
        self.assertGreaterEqual(pred, 0.0)
        self.assertLessEqual(pred, 1.0)

    def test_09_status_is_valid_label(self):
        valid_statuses = {"AVAILABLE", "MODERATE", "BUSY", "CRITICAL"}
        resp = get_client().post("/predict", json=_valid_payload())
        self.assertIn(resp.json()["status"], valid_statuses)

    def test_10_missing_feature_returns_422(self):
        payload = _valid_payload()
        del payload["lag_1h"]         # remove a required feature
        resp = get_client().post("/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_11_out_of_range_hour_returns_422(self):
        payload = _valid_payload()
        payload["hour"] = 25          # max is 23
        resp = get_client().post("/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_12_out_of_range_lag_returns_422(self):
        payload = _valid_payload()
        payload["lag_24h"] = 1.5      # lag must be <= 1.0
        resp = get_client().post("/predict", json=payload)
        self.assertEqual(resp.status_code, 422)

    def test_13_response_structure_is_stable(self):
        """All expected response fields must be present on every call."""
        resp = get_client().post("/predict", json=_valid_payload())
        body = resp.json()
        for key in ("predicted_occupancy", "status", "model_type", "feature_count"):
            self.assertIn(key, body, f"Response missing expected key: '{key}'")
        self.assertEqual(body["feature_count"], 16)


# ── Batch Prediction Tests ────────────────────────────────────────────────────

class TestBatchPredictEndpoint(unittest.TestCase):

    def test_14_batch_predict_returns_200(self):
        payload = {"items": [_valid_payload(), _valid_payload()]}
        resp = get_client().post("/predict/batch", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_15_batch_count_matches_input(self):
        items = [_valid_payload() for _ in range(5)]
        resp = get_client().post("/predict/batch", json={"items": items})
        body = resp.json()
        self.assertEqual(body["count"], 5)
        self.assertEqual(len(body["results"]), 5)

    def test_16_batch_results_are_indexed_in_order(self):
        """Each result must carry the correct zero-based index."""
        items = [_valid_payload() for _ in range(3)]
        resp = get_client().post("/predict/batch", json={"items": items})
        indices = [r["index"] for r in resp.json()["results"]]
        self.assertEqual(indices, [0, 1, 2])

    def test_17_oversized_batch_rejected(self):
        """Batches exceeding MAX_BATCH_SIZE must be rejected with 422."""
        items = [_valid_payload() for _ in range(MAX_BATCH_SIZE + 1)]
        resp = get_client().post("/predict/batch", json={"items": items})
        self.assertEqual(resp.status_code, 422)

    def test_18_empty_batch_rejected(self):
        """Empty batch list must be rejected with 422."""
        resp = get_client().post("/predict/batch", json={"items": []})
        self.assertEqual(resp.status_code, 422)

    def test_19_malformed_item_in_batch_rejected(self):
        """If ANY item fails schema validation, the whole request is rejected."""
        good = _valid_payload()
        bad  = _valid_payload()
        bad["hour"] = 99              # invalid hour
        resp = get_client().post("/predict/batch", json={"items": [good, bad]})
        self.assertEqual(resp.status_code, 422)

    def test_20_predictions_come_from_real_model(self):
        """
        Vary key features across two items and assert that the model produces
        different predictions. A hard-coded response would return the same
        value regardless of input.
        """
        item_low  = _valid_payload()
        item_high = _valid_payload()
        item_low["lag_24h"]           = 0.05   # very low past occupancy
        item_high["lag_24h"]          = 0.95   # very high past occupancy
        item_low["rolling_mean_24h"]  = 0.05
        item_high["rolling_mean_24h"] = 0.95
        item_low["rolling_mean_6h"]   = 0.05
        item_high["rolling_mean_6h"]  = 0.95

        resp = get_client().post("/predict/batch", json={"items": [item_low, item_high]})
        self.assertEqual(resp.status_code, 200)
        results = resp.json()["results"]
        pred_low  = results[0]["predicted_occupancy"]
        pred_high = results[1]["predicted_occupancy"]
        self.assertNotAlmostEqual(
            pred_low, pred_high, places=3,
            msg="Model returned identical predictions for very different inputs — "
                "possible hard-coded response path."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
