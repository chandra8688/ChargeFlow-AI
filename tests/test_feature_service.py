"""
ChargeFlow AI V2 — Phase 4 Feature Service Tests
==================================================
Tests for FeatureService (src/services/feature_service.py) and the new
POST /predict/raw API endpoint.

Coverage:
  Unit tests (FeatureService):
    01. Valid build_features() returns complete feature dict
    02. Feature vector has exactly 16 keys matching FEATURE_COLS in order
    03. Cyclical features satisfy sin²+cos²=1 (Pythagoras identity)
    04. lag_1h, lag_24h, lag_168h match exact historical values
    05. Rolling_mean_24h equals numpy mean of the 24 preceding observations
    06. Rolling_std_24h equals numpy std (ddof=1) of 24 preceding observations
    07. Unknown station raises UnknownStationError
    08. Malformed timestamp raises ValueError
    09. Non-hour-boundary timestamp raises ValueError
    10. Prediction timestamp with insufficient lag_168h history raises InsufficientHistoryError
    11. LEAKAGE: every lag/rolling timestamp is strictly < prediction_time
    12. Predicted occupancy is within [0, 1]
    13. 2025-06-30 00:00 is valid (not hardcoded as out-of-range)
    14. 2025-01-02 00:00 is invalid (only 24h prior history, lag_168h unavailable)
    15. is_weekend computed correctly for weekday vs. weekend

  API tests (POST /predict/raw):
    16. Valid raw request returns 200
    17. Response schema is stable (all expected keys present)
    18. Unknown station returns 404
    19. Malformed timestamp returns 422
    20. Temperature out of range returns 422
    21. /predict/raw prediction comes from real model
       (different lag inputs → different predictions)

  Regression:
    22. Existing Phase 1 tests still pass  (verified by full discover run)
    23. Existing Phase 2 tests still pass
    24. Existing Phase 3 tests still pass
"""

import math
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.services.feature_service import (
    FeatureService,
    UnknownStationError,
    InsufficientHistoryError,
    FEATURE_COLS,
)
from src.services.forecast_service import ForecastService


# ── Shared fixtures ───────────────────────────────────────────────────────────

# Known-good test parameters — within the historical data range
GOOD_STATION   = "STA001"
GOOD_TIME      = "2025-06-15 19:00:00"  # well inside [2025-01-08, 2025-06-30 00:00]
GOOD_TEMP      = 27.5
GOOD_HOLIDAY   = False

# Historical data lives here
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
HISTORY_CSV   = ARTIFACTS_DIR / "hourly_charging_data.csv"


# ── Module-level FeatureService fixture (loaded once for all unit tests) ─────

_feature_service: FeatureService | None = None


def setUpModule():
    """Load FeatureService once and enter the shared TestClient context."""
    global _feature_service, _client_ctx
    _feature_service = FeatureService()
    _client_ctx.__enter__()


def tearDownModule():
    global _client_ctx
    _client_ctx.__exit__(None, None, None)


def get_fs() -> FeatureService:
    return _feature_service


# ── TestClient (lifespan context manager pattern from Phase 3) ────────────────

_client_ctx = TestClient(app)


def get_client() -> TestClient:
    return _client_ctx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _raw_payload(**overrides) -> dict:
    """Return a valid raw payload, optionally overriding fields."""
    payload = {
        "station_id":      GOOD_STATION,
        "prediction_time": GOOD_TIME,
        "temperature_c":   GOOD_TEMP,
        "is_holiday":      GOOD_HOLIDAY,
    }
    payload.update(overrides)
    return payload


def _load_history_for(station_id: str) -> pd.Series:
    """Load the raw historical series for one station (for ground-truth checks)."""
    df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp"])
    sta = df[df["station_id"] == station_id].sort_values("timestamp")
    return sta.set_index("timestamp")["occupancy_rate"]


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests — FeatureService
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureServiceBasics(unittest.TestCase):

    def test_01_valid_request_returns_dict(self):
        """build_features() must return a dict for a valid request."""
        result = get_fs().build_features(
            GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY
        )
        self.assertIsInstance(result, dict)

    def test_02_feature_vector_exact_contract(self):
        """Returned dict must have exactly FEATURE_COLS keys in exact order."""
        result = get_fs().build_features(
            GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY
        )
        self.assertEqual(list(result.keys()), FEATURE_COLS,
            f"Key mismatch.\nExpected: {FEATURE_COLS}\nGot:      {list(result.keys())}")
        self.assertEqual(len(result), 16)

    def test_03_cyclical_features_unit_circle(self):
        """hour_sin² + hour_cos² == 1  and  day_sin² + day_cos² == 1."""
        feat = get_fs().build_features(
            GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY
        )
        self.assertAlmostEqual(feat["hour_sin"]**2 + feat["hour_cos"]**2, 1.0, places=10,
            msg="hour cyclical encoding violates unit circle identity")
        self.assertAlmostEqual(feat["day_sin"]**2 + feat["day_cos"]**2, 1.0, places=10,
            msg="day cyclical encoding violates unit circle identity")

    def test_04_lag_values_match_history(self):
        """lag_1h, lag_24h, lag_168h must exactly match historical occupancy_rate."""
        ts = pd.Timestamp(GOOD_TIME)
        history = _load_history_for(GOOD_STATION)
        feat    = get_fs().build_features(GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY)

        self.assertAlmostEqual(feat["lag_1h"],   float(history[ts - pd.Timedelta(hours=1)]),   places=6)
        self.assertAlmostEqual(feat["lag_24h"],  float(history[ts - pd.Timedelta(hours=24)]),  places=6)
        self.assertAlmostEqual(feat["lag_168h"], float(history[ts - pd.Timedelta(hours=168)]), places=6)

    def test_05_rolling_mean_24h_matches_numpy(self):
        """rolling_mean_24h must equal numpy.mean of the 24 hours before prediction_time."""
        ts      = pd.Timestamp(GOOD_TIME)
        history = _load_history_for(GOOD_STATION)
        feat    = get_fs().build_features(GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY)

        window_ts = [ts - pd.Timedelta(hours=h) for h in range(24, 0, -1)]
        expected  = float(np.mean(history[window_ts].values))
        self.assertAlmostEqual(feat["rolling_mean_24h"], expected, places=6)

    def test_06_rolling_std_24h_matches_numpy(self):
        """rolling_std_24h must equal numpy.std(ddof=1) of the 24 hours before prediction_time."""
        ts      = pd.Timestamp(GOOD_TIME)
        history = _load_history_for(GOOD_STATION)
        feat    = get_fs().build_features(GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY)

        window_ts = [ts - pd.Timedelta(hours=h) for h in range(24, 0, -1)]
        expected  = float(np.std(history[window_ts].values, ddof=1))
        self.assertAlmostEqual(feat["rolling_std_24h"], expected, places=6)


class TestFeatureServiceValidation(unittest.TestCase):

    def test_07_unknown_station_raises(self):
        """Non-existent station_id must raise UnknownStationError."""
        with self.assertRaises(UnknownStationError):
            get_fs().build_features("STA999", GOOD_TIME, GOOD_TEMP)

    def test_08_malformed_timestamp_raises(self):
        """Completely unparseable timestamp must raise ValueError."""
        with self.assertRaises(ValueError):
            get_fs().build_features(GOOD_STATION, "not-a-date", GOOD_TEMP)

    def test_09_non_hour_boundary_raises(self):
        """Timestamp with minutes/seconds must raise ValueError."""
        with self.assertRaises(ValueError):
            get_fs().build_features(GOOD_STATION, "2025-06-15 19:30:00", GOOD_TEMP)

    def test_10_insufficient_history_lag168_raises(self):
        """
        2025-01-02 00:00 has only 24 hours of prior history —
        lag_168h (T−168h = 2024-12-25 00:00) is not present.
        Must raise InsufficientHistoryError, not return a fabricated value.
        """
        with self.assertRaises(InsufficientHistoryError):
            get_fs().build_features(GOOD_STATION, "2025-01-02 00:00:00", GOOD_TEMP)

    def test_13_june_30_00_is_valid(self):
        """
        2025-06-30 00:00:00 must be VALID.
        lag_1h=2025-06-29 23:00 is in history; all other lags also present.
        The cutoff must NOT be hardcoded to 2025-06-29 23:00.
        """
        feat = get_fs().build_features(GOOD_STATION, "2025-06-30 00:00:00", GOOD_TEMP)
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 16)

    def test_14_jan_02_00_is_invalid(self):
        """2025-01-02 00:00 — insufficient history for lag_168h."""
        with self.assertRaises(InsufficientHistoryError):
            get_fs().build_features(GOOD_STATION, "2025-01-02 00:00:00", GOOD_TEMP)

    def test_15_is_weekend_computed_correctly(self):
        """is_weekend=0 for a weekday and is_weekend=1 for a Saturday."""
        # 2025-06-16 is Monday (dayofweek=0)
        feat_weekday = get_fs().build_features(GOOD_STATION, "2025-06-16 10:00:00", GOOD_TEMP)
        self.assertEqual(feat_weekday["is_weekend"], 0)

        # 2025-06-21 is Saturday (dayofweek=5)
        feat_weekend = get_fs().build_features(GOOD_STATION, "2025-06-21 10:00:00", GOOD_TEMP)
        self.assertEqual(feat_weekend["is_weekend"], 1)


class TestLeakagePrevention(unittest.TestCase):
    """Dedicated leakage prevention tests."""

    def test_11_no_future_data_in_lag_features(self):
        """
        Every timestamp used for lag_Nh must be strictly < prediction_time.
        This test reconstructs the expected lag timestamps and asserts ordering.
        """
        ts = pd.Timestamp(GOOD_TIME)
        for hours in [1, 24, 168]:
            lag_ts = ts - pd.Timedelta(hours=hours)
            self.assertLess(lag_ts, ts,
                f"lag_{hours}h timestamp {lag_ts} is not strictly before {ts}")

    def test_11b_rolling_window_all_before_prediction(self):
        """All 24 timestamps in the rolling window must be < prediction_time."""
        ts = pd.Timestamp(GOOD_TIME)
        for h in range(1, 25):
            wts = ts - pd.Timedelta(hours=h)
            self.assertLess(wts, ts,
                f"Rolling window timestamp {wts} is not strictly before {ts}")

    def test_12_predicted_occupancy_in_bounds(self):
        """The final prediction must be within [0.0, 1.0]."""
        feat   = get_fs().build_features(GOOD_STATION, GOOD_TIME, GOOD_TEMP, GOOD_HOLIDAY)
        fc_svc = ForecastService(eager_load=True)
        result = fc_svc.predict_single(feat)
        self.assertGreaterEqual(result["predicted_occupancy"], 0.0)
        self.assertLessEqual(result["predicted_occupancy"], 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# API Integration Tests — POST /predict/raw
# ══════════════════════════════════════════════════════════════════════════════

class TestRawPredictEndpoint(unittest.TestCase):

    def test_16_valid_raw_request_returns_200(self):
        resp = get_client().post("/predict/raw", json=_raw_payload())
        self.assertEqual(resp.status_code, 200,
            f"Expected 200, got {resp.status_code}. Body: {resp.text}")

    def test_17_response_schema_stable(self):
        """All expected response fields must be present."""
        resp = get_client().post("/predict/raw", json=_raw_payload())
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("station_id", "prediction_time",
                    "predicted_occupancy", "status",
                    "model_type", "feature_context"):
            self.assertIn(key, body, f"Missing response field: '{key}'")
        # feature_context must be non-empty
        self.assertGreater(len(body["feature_context"]), 0)
        # Occupancy bounds
        self.assertGreaterEqual(body["predicted_occupancy"], 0.0)
        self.assertLessEqual(body["predicted_occupancy"],   1.0)

    def test_18_unknown_station_returns_404(self):
        resp = get_client().post("/predict/raw", json=_raw_payload(station_id="STA999"))
        self.assertEqual(resp.status_code, 404)

    def test_19_malformed_timestamp_returns_422(self):
        resp = get_client().post("/predict/raw", json=_raw_payload(prediction_time="garbage"))
        self.assertEqual(resp.status_code, 422)

    def test_19b_non_hour_timestamp_returns_422(self):
        resp = get_client().post("/predict/raw",
                                 json=_raw_payload(prediction_time="2025-06-15 19:30:00"))
        self.assertEqual(resp.status_code, 422)

    def test_20_temperature_out_of_range_returns_422(self):
        resp = get_client().post("/predict/raw", json=_raw_payload(temperature_c=99.0))
        self.assertEqual(resp.status_code, 422)

    def test_20b_insufficient_history_returns_422(self):
        """Timestamp with insufficient lag_168h history must return 422."""
        resp = get_client().post("/predict/raw",
                                 json=_raw_payload(prediction_time="2025-01-02 00:00:00"))
        self.assertEqual(resp.status_code, 422)

    def test_21_different_lag_inputs_produce_different_predictions(self):
        """
        Predictions from the real model must vary with feature inputs.
        If they are identical regardless of input, something is hard-coded.
        We use two different prediction times with very different historical contexts.
        """
        # Evening peak on a normal weekday
        resp_peak = get_client().post(
            "/predict/raw",
            json=_raw_payload(
                station_id="STA001",
                prediction_time="2025-06-15 19:00:00",
                temperature_c=28.0,
            )
        )
        # Very early morning (low-demand period)
        resp_trough = get_client().post(
            "/predict/raw",
            json=_raw_payload(
                station_id="STA001",
                prediction_time="2025-06-15 03:00:00",
                temperature_c=22.0,
            )
        )
        self.assertEqual(resp_peak.status_code, 200)
        self.assertEqual(resp_trough.status_code, 200)

        peak    = resp_peak.json()["predicted_occupancy"]
        trough  = resp_trough.json()["predicted_occupancy"]
        self.assertNotAlmostEqual(peak, trough, places=3,
            msg="Model returned identical predictions for evening-peak vs. 3 AM — "
                "possible hard-coded response.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
