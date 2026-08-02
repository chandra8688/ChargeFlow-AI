"""
ChargeFlow AI V2 — DecisionService Unit & Integration Tests
============================================================
Comprehensive test suite for Phase 7 AI Decision Engine:
  - ForecastService & FeatureService integration
  - Candidate selection & charger compatibility policy
  - Haversine distance calculation
  - Deterministic ranking policy
  - Decision policy classification (STAY, REROUTE, NO_BETTER_ALTERNATIVE)
  - Occupancy bounding & self-exclusion
  - Offline execution without RAG / LLM keys
  - API endpoint integration (/recommend)
"""

import unittest
from fastapi.testclient import TestClient

from src.services.forecast_service import ForecastService
from src.services.feature_service import FeatureService, UnknownStationError, InsufficientHistoryError
from src.services.explainability_service import ExplainabilityService
from src.services.decision_service import (
    DecisionService,
    BUSY_THRESHOLD,
    MIN_OCCUPANCY_IMPROVEMENT,
    haversine_km,
)
from src.rag.llm_provider import MockLLMProvider
from src.rag.rag_service import RAGService
from src.api.main import app


class TestHaversineAndHelpers(unittest.TestCase):

    def test_01_haversine_same_point_is_zero(self):
        dist = haversine_km(12.937692, 77.719605, 12.937692, 77.719605)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_02_haversine_bengaluru_distance(self):
        # STA001 (12.937692, 77.719605) to STA006 (12.886034, 77.540307) ~19.9 km
        dist = haversine_km(12.937692, 77.719605, 12.886034, 77.540307)
        self.assertGreater(dist, 10.0)
        self.assertLess(dist, 30.0)


class TestDecisionServiceCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.forecast_svc = ForecastService(eager_load=True)
        cls.feature_svc = FeatureService()
        cls.explain_svc = ExplainabilityService(cls.forecast_svc)
        cls.rag_svc = RAGService(llm_provider=MockLLMProvider())
        cls.rag_svc.initialize()

        cls.dec_svc = DecisionService(
            forecast_service=cls.forecast_svc,
            feature_service=cls.feature_svc,
            explainability_service=cls.explain_svc,
            rag_service=cls.rag_svc,
        )

    def test_03_target_prediction_uses_real_forecast_service(self):
        res = self.dec_svc.recommend(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        self.assertEqual(res["selected_station"]["station_id"], "STA001")
        self.assertIn("predicted_occupancy", res["selected_station"])
        occ = res["selected_station"]["predicted_occupancy"]
        self.assertGreaterEqual(occ, 0.0)
        self.assertLessEqual(occ, 1.0)

    def test_04_target_excluded_from_alternatives(self):
        res = self.dec_svc.recommend(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        alt_ids = [alt["station_id"] for alt in res["alternatives"]]
        self.assertNotIn("STA001", alt_ids)

    def test_05_candidates_restricted_to_same_city(self):
        res = self.dec_svc.recommend(
            station_id="STA001",  # Bengaluru
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        for alt in res["alternatives"]:
            self.assertEqual(alt["city"], "Bengaluru")

    def test_06_deterministic_ranking_lowest_occupancy_first(self):
        res = self.dec_svc.recommend(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        alts = res["alternatives"]
        if len(alts) > 1:
            for i in range(len(alts) - 1):
                self.assertLessEqual(alts[i]["predicted_occupancy"], alts[i + 1]["predicted_occupancy"])

    def test_07_stay_decision_when_below_busy_threshold(self):
        # STA039 in Pune at 2025-06-15 19:00 has predicted occupancy = 0.6960 < 0.70
        res = self.dec_svc.recommend(
            station_id="STA039",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        self.assertLess(res["selected_station"]["predicted_occupancy"], BUSY_THRESHOLD)
        self.assertEqual(res["recommendation"], "STAY")

    def test_08_reroute_decision_when_meaningful_improvement_exists(self):
        # STA003 at 2025-06-15 19:00 has predicted occupancy = 0.9643 (CRITICAL), top alt STA004 = 0.7669 (imp = 0.1974 >= 0.10)
        res = self.dec_svc.recommend(
            station_id="STA003",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        self.assertGreaterEqual(res["selected_station"]["predicted_occupancy"], BUSY_THRESHOLD)
        self.assertEqual(res["recommendation"], "REROUTE")
        self.assertIsNotNone(res["top_alternative"])
        self.assertGreaterEqual(res["top_alternative"]["occupancy_improvement"], MIN_OCCUPANCY_IMPROVEMENT)

    def test_09_no_better_alternative_when_improvement_insufficient(self):
        # STA002 at 2025-06-15 19:00 has predicted occupancy = 0.7866, best alt = 0.7339 (imp = 0.0527 < 0.10)
        res = self.dec_svc.recommend(
            station_id="STA002",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
        )
        self.assertGreaterEqual(res["selected_station"]["predicted_occupancy"], BUSY_THRESHOLD)
        self.assertEqual(res["recommendation"], "NO_BETTER_ALTERNATIVE")

    def test_10_works_offline_without_rag(self):
        offline_dec_svc = DecisionService(
            forecast_service=self.forecast_svc,
            feature_service=self.feature_svc,
            explainability_service=self.explain_svc,
            rag_service=None,  # No RAG Service
        )
        res = offline_dec_svc.recommend(
            station_id="STA001",
            prediction_time="2025-06-15 19:00:00",
            temperature_c=28.0,
            include_rag_context=False,
        )
        self.assertIn("recommendation", res)
        self.assertIsNone(res["rag_context"])

    def test_11_unknown_station_raises_error(self):
        with self.assertRaises(UnknownStationError):
            self.dec_svc.recommend("STA999", "2025-06-15 19:00:00", 28.0)

    def test_12_insufficient_history_raises_error(self):
        with self.assertRaises(InsufficientHistoryError):
            self.dec_svc.recommend("STA001", "2025-01-02 12:00:00", 28.0)


class TestRecommendEndpointAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_13_post_recommend_returns_200_stay(self):
        resp = self.client.post("/recommend", json={
            "station_id": "STA039",
            "prediction_time": "2025-06-15 19:00:00",
            "temperature_c": 28.0,
            "is_holiday": False,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["selected_station"]["station_id"], "STA039")
        self.assertEqual(data["recommendation"], "STAY")

    def test_14_post_recommend_returns_200_reroute(self):
        resp = self.client.post("/recommend", json={
            "station_id": "STA003",
            "prediction_time": "2025-06-15 19:00:00",
            "temperature_c": 28.0,
            "is_holiday": False,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["selected_station"]["station_id"], "STA003")
        self.assertEqual(data["recommendation"], "REROUTE")
        self.assertIsNotNone(data["top_alternative"])

    def test_15_post_recommend_unknown_station_returns_404(self):
        resp = self.client.post("/recommend", json={
            "station_id": "STA999",
            "prediction_time": "2025-06-15 19:00:00",
        })
        self.assertEqual(resp.status_code, 404)

    def test_16_post_recommend_insufficient_history_returns_422(self):
        resp = self.client.post("/recommend", json={
            "station_id": "STA001",
            "prediction_time": "2025-01-02 12:00:00",
        })
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
