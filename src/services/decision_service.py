from __future__ import annotations

"""
ChargeFlow AI V2 — AI Decision & Recommendation Engine
========================================================
Orchestrates real machine-learning forecasts, explainability diagnostics,
and optional grounded RAG context into actionable EV charging decisions.

Decision Policy (Configurable Business/Policy Thresholds — NOT learned ML parameters):
  - BUSY_THRESHOLD = 0.70
  - MIN_OCCUPANCY_IMPROVEMENT = 0.10

Logic:
  1. Evaluate Target Station:
     - FeatureService.build_features() -> ForecastService.predict_single()
  2. Candidate Alternatives:
     - Filter data/stations.csv by:
         same city AND compatible charger standard AND station_id != target
     - For each candidate at the SAME timestamp:
         FeatureService.build_features() -> ForecastService.predict_single()
         Calculate Haversine distance from target station
  3. Deterministic Ranking Policy:
     - Primary: predicted_occupancy ASCENDING (lowest predicted demand wins)
     - Secondary: distance_km ASCENDING (closest station wins as tie-breaker)
     - Tertiary: station_id ASCENDING (deterministic tie-breaker)
  4. Decision Policy Classification:
     - If target_occupancy < BUSY_THRESHOLD:
         recommendation = "STAY"
     - If target_occupancy >= BUSY_THRESHOLD:
         Calculate occupancy_improvement = target_occupancy - best_alternative_occupancy
         If best_alternative_occupancy < target_occupancy AND occupancy_improvement >= MIN_OCCUPANCY_IMPROVEMENT:
             recommendation = "REROUTE"
         Else:
             recommendation = "NO_BETTER_ALTERNATIVE"

5. Grounding & RAG:
     - RAG is strictly OPTIONAL and NEVER influences numerical rankings or decisions.
     - DecisionService functions 100% offline without API keys.
"""

import time
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from src.services.forecast_service import ForecastService
from src.services.feature_service import FeatureService, UnknownStationError, InsufficientHistoryError
from src.services.explainability_service import ExplainabilityService

# RAGService is imported lazily inside DecisionService methods to prevent eager PyTorch initialization
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.rag.rag_service import RAGService


# ── Decision Policy Constants (Business/Policy Thresholds — NOT ML parameters) ──
BUSY_THRESHOLD = 0.70
MIN_OCCUPANCY_IMPROVEMENT = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIONS_CSV = PROJECT_ROOT / "data" / "stations.csv"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two GPS coordinates."""
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def get_charger_standard(charger_type: str) -> str:
    """Extract standard family prefix for compatibility grouping."""
    ct = str(charger_type).upper()
    if "CCS2" in ct:
        return "CCS2"
    elif "TYPE 2" in ct:
        return "TYPE 2"
    elif "CHADEMO" in ct:
        return "CHADEMO"
    return ct


class DecisionService:
    """
    AI Decision & Recommendation Engine orchestrating ML forecasts and diagnostics.
    """

    def __init__(
        self,
        forecast_service: ForecastService,
        feature_service: FeatureService,
        explainability_service: Optional[ExplainabilityService] = None,
        rag_service: Optional[RAGService] = None,
        busy_threshold: float = BUSY_THRESHOLD,
        min_occupancy_improvement: float = MIN_OCCUPANCY_IMPROVEMENT,
    ):
        self.forecast_service = forecast_service
        self.feature_service = feature_service
        self.explainability_service = explainability_service
        self.rag_service = rag_service
        self.busy_threshold = busy_threshold
        self.min_occupancy_improvement = min_occupancy_improvement

        # Load static station metadata DataFrame
        if not STATIONS_CSV.exists():
            raise FileNotFoundError(f"Stations metadata file missing: {STATIONS_CSV}")
        self.stations_df = pd.read_csv(STATIONS_CSV)
        self.stations_df["standard"] = self.stations_df["charger_type"].apply(get_charger_standard)

    def recommend(
        self,
        station_id: str,
        prediction_time: str,
        temperature_c: float,
        is_holiday: bool = False,
        max_alternatives: int = 3,
        include_rag_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates ML-backed rerouting decision and alternative rankings.

        Returns structured result dict.
        """
        t0 = time.perf_counter()

        # Step 1: Validate target station
        target_rows = self.stations_df[self.stations_df["station_id"] == station_id]
        if target_rows.empty:
            raise UnknownStationError(f"Station ID '{station_id}' not found in network metadata.")
        target_meta = target_rows.iloc[0].to_dict()

        # Step 2: Target station ML prediction (raises InsufficientHistoryError or ValueError if invalid)
        target_features = self.feature_service.build_features(
            station_id=station_id,
            prediction_time=prediction_time,
            temperature_c=temperature_c,
            is_holiday=is_holiday,
        )
        target_pred = self.forecast_service.predict_single(target_features)
        target_occ = float(target_pred["predicted_occupancy"])
        target_status = str(target_pred["status"])

        # Step 3: Identify compatible candidate stations in the same city
        city = target_meta["city"]
        standard = target_meta["standard"]

        candidates = self.stations_df[
            (self.stations_df["city"] == city)
            & (self.stations_df["station_id"] != station_id)
            & (self.stations_df["standard"] == standard)
        ].copy()

        # Fallback: if no same-standard candidate in city, evaluate all other stations in same city
        if candidates.empty:
            candidates = self.stations_df[
                (self.stations_df["city"] == city)
                & (self.stations_df["station_id"] != station_id)
            ].copy()

        # Step 4: Batch forecast all candidate stations at the exact same prediction timestamp
        candidate_results = []
        for _, c_row in candidates.iterrows():
            c_id = c_row["station_id"]
            try:
                c_feat = self.feature_service.build_features(
                    station_id=c_id,
                    prediction_time=prediction_time,
                    temperature_c=temperature_c,
                    is_holiday=is_holiday,
                )
                c_pred = self.forecast_service.predict_single(c_feat)
                c_occ = float(c_pred["predicted_occupancy"])
                c_status = str(c_pred["status"])

                dist_km = haversine_km(
                    float(target_meta["latitude"]),
                    float(target_meta["longitude"]),
                    float(c_row["latitude"]),
                    float(c_row["longitude"]),
                )

                candidate_results.append({
                    "station_id": c_id,
                    "name": c_row["name"],
                    "city": c_row["city"],
                    "charger_type": c_row["charger_type"],
                    "predicted_occupancy": round(c_occ, 4),
                    "status": c_status,
                    "distance_km": round(dist_km, 2),
                    "occupancy_delta": round(c_occ - target_occ, 4),
                    "occupancy_improvement": round(target_occ - c_occ, 4),
                })
            except Exception:
                continue

        # Step 5: Deterministic Ranking Policy
        # 1. predicted_occupancy ASC
        # 2. distance_km ASC
        # 3. station_id ASC
        candidate_results.sort(
            key=lambda x: (x["predicted_occupancy"], x["distance_km"], x["station_id"])
        )

        ranked_alternatives = candidate_results[:max_alternatives]
        top_alt = ranked_alternatives[0] if ranked_alternatives else None

        # Step 6: Decision Policy Classification
        if target_occ < self.busy_threshold:
            recommendation = "STAY"
            reason = (
                f"Selected station {station_id} has acceptable predicted occupancy "
                f"({target_occ*100:.1f}% < {self.busy_threshold*100:.0f}% policy threshold). "
                f"No rerouting required."
            )
        else:
            if top_alt is not None:
                best_occ = top_alt["predicted_occupancy"]
                improvement = target_occ - best_occ

                if best_occ < target_occ and improvement >= self.min_occupancy_improvement:
                    recommendation = "REROUTE"
                    reason = (
                        f"Selected station {station_id} has high predicted occupancy ({target_occ*100:.1f}%). "
                        f"Recommended reroute to {top_alt['station_id']} in {city} with "
                        f"lower predicted occupancy ({best_occ*100:.1f}%, improvement: {improvement*100:.1f}% >= "
                        f"{self.min_occupancy_improvement*100:.0f}% policy threshold)."
                    )
                else:
                    recommendation = "NO_BETTER_ALTERNATIVE"
                    reason = (
                        f"Selected station {station_id} has high predicted occupancy ({target_occ*100:.1f}%), "
                        f"but best alternative {top_alt['station_id']} offers insufficient occupancy improvement "
                        f"({improvement*100:.1f}% < {self.min_occupancy_improvement*100:.0f}% policy threshold)."
                    )
            else:
                recommendation = "NO_BETTER_ALTERNATIVE"
                reason = f"No compatible candidate alternative stations found in {city}."

        # Step 7: Explainability diagnostics for target station
        dispersion = None
        top_ctx = None
        if self.explainability_service is not None:
            dispersion = self.explainability_service.tree_dispersion(target_features)
            top_ctx = self.explainability_service.top_n_feature_context(target_features, n=5)

        # Step 8: Optional RAG context (never influences recommendation logic)
        rag_output = None
        if include_rag_context and self.rag_service is not None:
            rag_query_str = (
                f"How should an operator handle a {target_status} occupancy prediction at an EV charging station?"
            )
            rag_output = self.rag_service.query(rag_query_str, top_k=2)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "selected_station": {
                "station_id": station_id,
                "name": target_meta["name"],
                "city": city,
                "charger_type": target_meta["charger_type"],
                "predicted_occupancy": round(target_occ, 4),
                "status": target_status,
            },
            "recommendation": recommendation,
            "recommendation_reason": reason,
            "policy_thresholds": {
                "busy_threshold": self.busy_threshold,
                "min_occupancy_improvement": self.min_occupancy_improvement,
            },
            "top_alternative": top_alt,
            "alternatives": ranked_alternatives,
            "tree_dispersion": dispersion,
            "top_feature_context": top_ctx,
            "rag_context": rag_output,
            "latency_ms": round(latency_ms, 2),
        }
