"""
ChargeFlow AI — Station Recommender
=====================================
Multi-criteria weighted scoring system to recommend the optimal
EV charging station for a given driver's context.

Design Philosophy:
  - Transparent scoring (no black box) — each criterion is explicit
  - Easily explainable to judges and end users
  - Adjustable weights allow operator customization
  - Fast: O(n_stations) per query — scales to 10,000+ stations
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


# ── Haversine Distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two GPS coordinates."""
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


# ── Charger Compatibility ─────────────────────────────────────────────────────

VEHICLE_CHARGER_COMPAT = {
    "Tata Nexon EV":      ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)", "DC CCS2 (50 kW)"],
    "MG ZS EV":           ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)", "DC CCS2 (50 kW)"],
    "Hyundai Kona Electric": ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)", "DC CCS2 (50 kW)"],
    "Ather 450X":         ["AC Type 2 (7.4 kW)"],
    "Ola S1 Pro":         ["AC Type 2 (7.4 kW)"],
    "Tata Tigor EV":      ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)"],
    "Mahindra XUV400":    ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)", "DC CCS2 (50 kW)"],
    "BYD Atto 3":         ["AC Type 2 (22 kW)", "DC CCS2 (50 kW)", "DC CCS2 (150 kW)"],
    "Kia EV6":            ["AC Type 2 (22 kW)", "DC CCS2 (150 kW)"],
    "BMW iX":             ["AC Type 2 (22 kW)", "DC CCS2 (150 kW)"],
}

DEFAULT_CHARGER_TYPES = ["AC Type 2 (7.4 kW)", "AC Type 2 (22 kW)"]  # Fallback


# ── Recommender ───────────────────────────────────────────────────────────────

class StationRecommender:
    """
    Recommends top-K charging stations using multi-criteria scoring.

    Scoring Formula:
        score = w_dist    * proximity_score   (inverse distance)
              + w_avail   * availability_score (available slots ratio)
              + w_wait    * wait_score         (inverse wait time)
              + w_amenity * amenity_score      (normalised amenities)
              + w_tariff  * cost_score         (inverse tariff)

    All sub-scores are normalised to [0, 1] before weighting.
    """

    DEFAULT_WEIGHTS = {
        "distance":    0.35,  # Most important: proximity
        "availability": 0.30,  # Second: can I plug in now?
        "wait_time":   0.20,  # Third: how long to wait?
        "amenities":   0.10,  # Comfort matters but secondary
        "tariff":      0.05,  # Cost matters least in urgent charge
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize a pandas Series to [0, 1]."""
        min_val, max_val = series.min(), series.max()
        if max_val == min_val:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - min_val) / (max_val - min_val)

    def recommend(self,
                  user_lat: float,
                  user_lon: float,
                  vehicle_type: str,
                  realtime_df: pd.DataFrame,
                  stations_df: pd.DataFrame,
                  max_distance_km: float = 25.0,
                  top_k: int = 3) -> pd.DataFrame:
        """
        Recommend top-K stations for a given user.

        Args:
            user_lat        : User's current latitude
            user_lon        : User's current longitude
            vehicle_type    : EV model name (for charger compatibility)
            realtime_df     : Current station status DataFrame
            stations_df     : Static station metadata DataFrame
            max_distance_km : Search radius in km
            top_k           : Number of recommendations to return

        Returns:
            DataFrame with ranked stations and score breakdown
        """
        # Merge realtime + static metadata (exclude total_slots — already in realtime_df)
        df = realtime_df.merge(
            stations_df[["station_id", "amenities_score", "tariff_per_kwh",
                          "is_24x7", "has_fast_charger"]],
            on="station_id", how="left"
        )

        # Filter by charger compatibility
        compatible_types = VEHICLE_CHARGER_COMPAT.get(vehicle_type, DEFAULT_CHARGER_TYPES)
        df = df[df["charger_type"].isin(compatible_types)].copy()

        if df.empty:
            return pd.DataFrame(columns=["station_id", "name", "final_score", "rank"])

        # Compute distance for each station
        df["distance_km"] = df.apply(
            lambda r: haversine_km(user_lat, user_lon, r["latitude"], r["longitude"]),
            axis=1,
        )

        # Filter by max radius
        df = df[df["distance_km"] <= max_distance_km].copy()

        if df.empty:
            return pd.DataFrame(columns=["station_id", "name", "final_score", "rank"])

        # Compute individual sub-scores (higher = better)
        df["proximity_score"]    = 1 / (1 + df["distance_km"])         # inverse distance
        df["availability_score"] = df["available_slots"] / df["total_slots"].clip(lower=1)
        df["wait_score"]         = 1 / (1 + df["estimated_wait_mins"]) # inverse wait
        df["amenity_score"]      = df["amenities_score"] / 10.0         # already 0–10
        df["cost_score"]         = 1 / (1 + df["tariff_per_kwh"])       # inverse cost

        # Normalize all sub-scores
        for col in ["proximity_score", "availability_score", "wait_score",
                    "amenity_score", "cost_score"]:
            df[col] = self._normalize(df[col])

        # Compute weighted final score
        w = self.weights
        df["final_score"] = (
            w["distance"]    * df["proximity_score"]    +
            w["availability"] * df["availability_score"] +
            w["wait_time"]   * df["wait_score"]          +
            w["amenities"]   * df["amenity_score"]       +
            w["tariff"]      * df["cost_score"]
        )

        # Rank and return top-K
        df = df.sort_values("final_score", ascending=False).head(top_k).copy()
        df["rank"] = range(1, len(df) + 1)

        output_cols = [
            "rank", "station_id", "name", "city", "operator",
            "charger_type", "distance_km", "available_slots",
            "estimated_wait_mins", "current_load_kw", "status",
            "final_score", "proximity_score", "availability_score",
            "wait_score", "amenity_score", "cost_score",
            "latitude", "longitude",
        ]
        return df[[c for c in output_cols if c in df.columns]].reset_index(drop=True)

    def explain(self, recommendation_row: pd.Series) -> str:
        """Generate a human-readable explanation for a recommendation."""
        reasons = []

        if recommendation_row.get("proximity_score", 0) >= 0.7:
            reasons.append(f"📍 Only {recommendation_row['distance_km']:.1f} km away")
        if recommendation_row.get("available_slots", 0) > 0:
            reasons.append(f"✅ {int(recommendation_row['available_slots'])} slots available now")
        if recommendation_row.get("estimated_wait_mins", 99) <= 5:
            reasons.append("⚡ Near-zero wait time")
        elif recommendation_row.get("wait_score", 0) >= 0.6:
            reasons.append(f"⏱️ Short wait: ~{recommendation_row['estimated_wait_mins']:.0f} mins")
        if recommendation_row.get("amenity_score", 0) >= 0.7:
            reasons.append("☕ High amenity score (cafe, WiFi, shade)")

        return " | ".join(reasons) if reasons else "Balanced score across all criteria"
