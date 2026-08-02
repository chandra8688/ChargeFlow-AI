"""
ChargeFlow AI V2 — Feature Engineering Service
================================================
Converts a raw station prediction request into the exact 16-feature vector
required by the Phase 2 DemandForecaster model.

Training-Serving Parity Contract:
  Every feature is computed with the IDENTICAL formula used in
  src/data/preprocessor.py (DataPreprocessor.engineer_features).
  Deviations would silently corrupt inference — this module is the
  single source of truth for online feature computation.

  Training formula → Online equivalent in this module:
  ─────────────────────────────────────────────────────────────────
  df["hour"] = df["timestamp"].dt.hour
      → ts.hour

  df["hour_sin"] = sin(2π·hour/24)
      → math.sin(2π·hour/24)         [identical]

  df["lag_1h"] = groupby(station).shift(1)
      → occupancy_rate at (T - 1h)   [exact timestamp lookup]

  rolling_mean_6h: hist_series = shift(1); rolling(6).mean()
      → mean of occupancy at {T-6h, T-5h, ..., T-1h}  [identical window]

Leakage Prevention:
  Every historical lookup targets timestamp < prediction_time.
  An assertion guards against any future-data contamination.

Memory Efficiency:
  Historical CSV is loaded once at construction into a per-station
  pd.Series indexed by timestamp. Subsequent requests do O(1) dict
  lookups — the CSV is never re-read.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.data.preprocessor import DataPreprocessor

# ── Feature contract — must match Phase 2 exactly ────────────────────────────
FEATURE_COLS: list = DataPreprocessor.FEATURE_COLS   # 16 features
TARGET_COL:   str  = DataPreprocessor.TARGET_COL     # "occupancy_rate"


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class UnknownStationError(ValueError):
    """station_id not found in the historical dataset."""


class InsufficientHistoryError(ValueError):
    """Not enough prior observations exist for the requested prediction_time."""


# ── Feature Service ───────────────────────────────────────────────────────────

class FeatureService:
    """
    Builds the 16-feature dict for a raw prediction request.

    The 18 MB historical CSV is loaded once at construction and split into
    per-station pd.Series objects keyed by timestamp.  All subsequent
    feature lookups are in-memory.
    """

    # Lag windows in hours — must stay aligned with DataPreprocessor
    _LAG_HOURS: Dict[str, int] = {
        "lag_1h":   1,
        "lag_24h":  24,
        "lag_168h": 168,
    }

    # Rolling windows: (feature_name, hours, aggregation)
    # Window = occupancy at {T-W, T-(W-1), ..., T-1h}  (all strictly < T)
    _ROLLING_DEFS = [
        ("rolling_mean_6h",  6,  "mean"),
        ("rolling_mean_24h", 24, "mean"),
        ("rolling_std_24h",  24, "std"),
    ]

    def __init__(
        self,
        history_path: Optional[Union[str, Path]] = None,
        stations_csv_path: Optional[Union[str, Path]] = None,
    ):
        _root = Path(__file__).resolve().parent.parent.parent

        if history_path is None:
            history_path = _root / "artifacts" / "hourly_charging_data.csv"
        if stations_csv_path is None:
            stations_csv_path = _root / "data" / "stations.csv"

        self._history_path  = Path(history_path)
        self._stations_path = Path(stations_csv_path)

        # Populated by _load()
        # _station_history: {station_id: pd.Series(index=timestamp, values=occupancy_rate)}
        self._station_history: Dict[str, pd.Series] = {}
        self._valid_station_ids: set = set()

        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load historical CSV once. Never called again after construction."""
        if not self._history_path.exists():
            raise FileNotFoundError(
                f"Historical data not found: {self._history_path}. "
                "Run 'python -m src.train_evaluate' to generate it."
            )

        df = pd.read_csv(self._history_path, parse_dates=["timestamp"])
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        # Build per-station Series: index=timestamp, value=occupancy_rate
        # This gives O(1) timestamp lookups via series[ts]
        for station_id, group in df.groupby("station_id"):
            self._station_history[station_id] = (
                group.set_index("timestamp")[TARGET_COL].sort_index()
            )

        self._valid_station_ids = set(self._station_history.keys())

    # ── Public helpers ────────────────────────────────────────────────────────

    def list_stations(self) -> list:
        """Return sorted list of all station IDs present in history."""
        return sorted(self._valid_station_ids)

    def history_bounds(self, station_id: str) -> Dict[str, pd.Timestamp]:
        """Return the min/max timestamps available for a station."""
        self._validate_station(station_id)
        s = self._station_history[station_id]
        return {"min": s.index.min(), "max": s.index.max()}

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate_station(self, station_id: str) -> None:
        if station_id not in self._valid_station_ids:
            raise UnknownStationError(
                f"Station '{station_id}' not found in historical data. "
                f"First 5 valid stations: {sorted(self._valid_station_ids)[:5]}."
            )

    @staticmethod
    def _parse_timestamp(
        prediction_time: Union[str, "datetime", pd.Timestamp]
    ) -> pd.Timestamp:
        """
        Parse to a tz-naive pd.Timestamp on an exact hour boundary.
        The Phase 2 model was trained on hourly data; sub-hour resolution
        has no meaning in the feature space.
        """
        try:
            ts = pd.Timestamp(prediction_time)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse prediction_time '{prediction_time}': {exc}"
            )

        if ts.tz is not None:
            ts = ts.tz_localize(None)

        ts_floored = ts.floor("h")
        if ts != ts_floored:
            raise ValueError(
                f"prediction_time must be on an exact hour boundary "
                f"(e.g. '2025-06-15 19:00:00'). "
                f"Received: {prediction_time}. "
                f"Nearest valid hour: {ts_floored}"
            )
        return ts

    # ── Private feature computation ───────────────────────────────────────────

    def _lookup_lag(
        self, history: pd.Series, ts: pd.Timestamp, hours: int
    ) -> float:
        """
        Retrieve occupancy_rate exactly `hours` before `ts`.

        LEAKAGE GUARD: target_ts = ts − hours < ts  (strict inequality).
        Raises InsufficientHistoryError if the exact timestamp is absent.
        """
        target_ts = ts - pd.Timedelta(hours=hours)

        # Assertion enforces leakage prevention — never future data
        assert target_ts < ts, (
            f"Leakage guard violation: lag target {target_ts} >= prediction_time {ts}"
        )

        if target_ts not in history.index:
            raise InsufficientHistoryError(
                f"lag_{hours}h for {ts} requires observation at {target_ts}, "
                f"which is not present in historical data. "
                f"History spans {history.index.min()} → {history.index.max()}."
            )
        return float(history[target_ts])

    def _compute_rolling(
        self,
        history: pd.Series,
        ts: pd.Timestamp,
        window_hours: int,
        agg: str,
    ) -> float:
        """
        Compute rolling aggregation over the `window_hours` observations
        immediately before `ts`:  {T − W·h, T − (W−1)·h, …, T − 1h}.

        This exactly replicates:
          hist_series = shift(1)           # → values at t-1
          rolling(W).mean()                # → mean of window ending at t-1

        LEAKAGE GUARD: every timestamp in the window is < ts.
        """
        # Build the window timestamps: T-W, T-(W-1), ..., T-1  (all < T)
        window_ts = [ts - pd.Timedelta(hours=h) for h in range(window_hours, 0, -1)]

        # Leakage assertion for every window element
        for wts in window_ts:
            assert wts < ts, (
                f"Leakage guard violation: rolling window timestamp {wts} >= {ts}"
            )

        missing = [wts for wts in window_ts if wts not in history.index]
        if missing:
            raise InsufficientHistoryError(
                f"rolling_{agg}_{window_hours}h for {ts}: "
                f"{len(missing)} required observations are missing "
                f"(earliest missing: {missing[0]}). "
                f"History spans {history.index.min()} → {history.index.max()}."
            )

        values = history[window_ts].values.astype(float)

        if agg == "mean":
            return float(np.mean(values))
        elif agg == "std":
            # ddof=1 matches pandas rolling().std() default
            return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        else:
            raise ValueError(f"Unknown aggregation function: '{agg}'")

    # ── Primary public method ─────────────────────────────────────────────────

    def build_features(
        self,
        station_id: str,
        prediction_time: Union[str, "datetime", pd.Timestamp],
        temperature_c: float,
        is_holiday: Union[bool, int] = False,
    ) -> Dict[str, Any]:
        """
        Build the complete 16-feature dict for the Phase 2 DemandForecaster.

        All features are computed with formulas identical to
        DataPreprocessor.engineer_features (Phase 2 training pipeline).
        All historical lookups enforce timestamp < prediction_time.

        Args:
            station_id:      Station identifier (must exist in historical data).
            prediction_time: ISO-format string or datetime, exact hour boundary.
            temperature_c:   Ambient temperature in Celsius [−5, 50].
            is_holiday:      Whether the prediction date is a public holiday.

        Returns:
            Dict[str, Any] with keys = FEATURE_COLS (16 entries, same order).

        Raises:
            UnknownStationError:      station not found.
            ValueError:               malformed timestamp or temp out of range.
            InsufficientHistoryError: required lag/rolling observation missing.
        """
        # Step 1 — Validate inputs
        self._validate_station(station_id)
        ts = self._parse_timestamp(prediction_time)

        if not (-5.0 <= float(temperature_c) <= 50.0):
            raise ValueError(
                f"temperature_c={temperature_c} is outside physical range [−5, 50]."
            )

        history = self._station_history[station_id]

        # Step 2 — Calendar features (identical to DataPreprocessor)
        hour        = ts.hour
        day_of_week = ts.dayofweek   # 0 = Monday … 6 = Sunday
        month       = ts.month

        # Step 3 — Cyclical encodings (identical to DataPreprocessor)
        hour_sin = math.sin(2.0 * math.pi * hour        / 24.0)
        hour_cos = math.cos(2.0 * math.pi * hour        / 24.0)
        day_sin  = math.sin(2.0 * math.pi * day_of_week /  7.0)
        day_cos  = math.cos(2.0 * math.pi * day_of_week /  7.0)

        # Step 4 — Boolean flags (identical to DataPreprocessor)
        is_weekend = int(day_of_week >= 5)
        is_holiday = int(bool(is_holiday))

        # Step 5 — Lag features (exact timestamp lookups, all < ts)
        lag_1h   = self._lookup_lag(history, ts, 1)
        lag_24h  = self._lookup_lag(history, ts, 24)
        lag_168h = self._lookup_lag(history, ts, 168)

        # Step 6 — Rolling features (window entirely < ts)
        rolling_mean_6h  = self._compute_rolling(history, ts, 6,  "mean")
        rolling_mean_24h = self._compute_rolling(history, ts, 24, "mean")
        rolling_std_24h  = self._compute_rolling(history, ts, 24, "std")

        # Step 7 — Assemble in EXACT FEATURE_COLS order
        feature_dict: Dict[str, Any] = {
            "hour":             hour,
            "day_of_week":      day_of_week,
            "month":            month,
            "hour_sin":         hour_sin,
            "hour_cos":         hour_cos,
            "day_sin":          day_sin,
            "day_cos":          day_cos,
            "is_weekend":       is_weekend,
            "is_holiday":       is_holiday,
            "temperature_c":    float(temperature_c),
            "lag_1h":           lag_1h,
            "lag_24h":          lag_24h,
            "lag_168h":         lag_168h,
            "rolling_mean_6h":  rolling_mean_6h,
            "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h":  rolling_std_24h,
        }

        # Parity check — keys must exactly match FEATURE_COLS in order
        assert list(feature_dict.keys()) == FEATURE_COLS, (
            f"Feature key mismatch! Expected {FEATURE_COLS}, "
            f"got {list(feature_dict.keys())}"
        )

        return feature_dict

    # ── Explainability context ────────────────────────────────────────────────

    def build_context(
        self,
        feature_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a human-readable feature context dict for explainability output.

        This is purely DESCRIPTIVE — it summarises what the model features contain.
        It does NOT explain causality and must NOT be described as a causal explanation.
        """
        hour       = feature_dict["hour"]
        lag_1h     = feature_dict["lag_1h"]
        lag_24h    = feature_dict["lag_24h"]
        lag_168h   = feature_dict["lag_168h"]
        roll_24h   = feature_dict["rolling_mean_24h"]

        # Describe the time-of-day period (purely calendar, not model output)
        if   7 <= hour <= 10:  period = "morning peak (7–10 AM)"
        elif 12 <= hour <= 14: period = "midday period (12–2 PM)"
        elif 18 <= hour <= 21: period = "evening peak (6–9 PM)"
        elif 0  <= hour <= 5:  period = "overnight off-peak (12–6 AM)"
        else:                   period = f"hour {hour:02d}:00"

        # Describe recent trend
        if   roll_24h >= 0.70: trend = "high"
        elif roll_24h >= 0.40: trend = "moderate"
        else:                   trend = "low"

        return {
            "time_period":       period,
            "lag_1h":            round(lag_1h,   4),
            "lag_24h":           round(lag_24h,  4),
            "lag_168h":          round(lag_168h, 4),
            "rolling_mean_24h":  round(roll_24h, 4),
            "recent_24h_trend":  trend,
            "is_weekend":        bool(feature_dict["is_weekend"]),
            "is_holiday":        bool(feature_dict["is_holiday"]),
        }
