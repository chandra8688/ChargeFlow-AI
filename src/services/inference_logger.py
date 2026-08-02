"""
ChargeFlow AI V2 — Inference Logger
=====================================
Records every inference event to an append-only JSON-Lines file.

Design:
  - Zero infrastructure dependencies (no Kafka, Redis, Prometheus, MLflow)
  - Append-only JSONL: human-readable, grep-able, pandas-loadable in one line
  - Thread-safe: a threading.Lock guards every write — no reliance on
    assumed atomic OS behaviour or Python GIL properties
  - Synchronous: log() completes before returning; no background threads

Log record schema:
  prediction_id        str  — uuid4, unique per inference event
  logged_at            str  — UTC ISO-8601 timestamp
  station_id           str  — station identifier
  prediction_time      str  — the prediction's target timestamp
  predicted_occupancy  float — model output, clipped to [0, 1]
  status               str  — AVAILABLE | MODERATE | BUSY | CRITICAL
  model_version        str  — trained_at from the model metadata artifact
                              (traces prediction to the exact saved model)
  inference_latency_ms float — wall-clock time for the inference call
  source               str  — "api" | "streamlit"
  key_features         dict — subset of input features for audit trail:
                              hour, lag_1h, lag_24h, lag_168h,
                              rolling_mean_24h, temperature_c

model_version is intentionally set to the artifact's trained_at timestamp,
not a separate version counter. This is the simplest trace without MLflow.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd


_KEY_FEATURE_NAMES = [
    "hour", "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "temperature_c",
]


class InferenceLogger:
    """
    Lightweight inference event logger.

    Thread-safe append-only JSONL writer.
    One instance should be shared across all inference calls within a
    process (via dependency injection or @st.cache_resource).
    """

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
    ):
        if log_path is None:
            log_path = (
                Path(__file__).resolve().parent.parent.parent
                / "logs"
                / "inference_log.jsonl"
            )
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Explicit lock — thread-safety must not rely on GIL or OS atomic writes
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        station_id:           str,
        prediction_time:      str,
        predicted_occupancy:  float,
        status:               str,
        model_version:        str,
        inference_latency_ms: float,
        source:               str,
        key_features:         Dict[str, Any],
    ) -> str:
        """
        Append one inference event to the log file.

        Args:
            station_id:           Station identifier.
            prediction_time:      Target timestamp for the prediction.
            predicted_occupancy:  Model output in [0, 1].
            status:               Status label from ForecastService.
            model_version:        trained_at from the model metadata artifact.
            inference_latency_ms: Wall-clock inference duration in ms.
            source:               "api" or "streamlit".
            key_features:         Subset of input features for the audit trail.

        Returns:
            prediction_id (str) — uuid4 unique identifier for this event.
        """
        prediction_id = str(uuid.uuid4())
        entry: Dict[str, Any] = {
            "prediction_id":        prediction_id,
            "logged_at":            datetime.now(timezone.utc).isoformat(),
            "station_id":           station_id,
            "prediction_time":      prediction_time,
            "predicted_occupancy":  round(float(predicted_occupancy), 4),
            "status":               status,
            "model_version":        model_version,
            "inference_latency_ms": round(float(inference_latency_ms), 2),
            "source":               source,
            "key_features":         {
                k: round(float(v), 6) if isinstance(v, float) else v
                for k, v in key_features.items()
            },
        }

        line = json.dumps(entry, ensure_ascii=False)
        # Lock around both open() and write() — no assumed OS atomicity
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        return prediction_id

    def read_log(self) -> pd.DataFrame:
        """
        Load the entire inference log as a DataFrame.

        Thread-safe: acquires the lock while reading.
        Returns an empty DataFrame if the log file does not yet exist.
        """
        if not self._log_path.exists():
            return pd.DataFrame()

        records = []
        with self._lock:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

        return pd.DataFrame(records) if records else pd.DataFrame()

    def log_path(self) -> Path:
        """Return the path to the current log file."""
        return self._log_path

    # ── Convenience helper ────────────────────────────────────────────────────

    @staticmethod
    def extract_key_features(feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the canonical key-feature subset from a full 16-feature dict.
        Used by both the API and Streamlit callers for consistent logging.
        """
        return {k: feature_dict[k] for k in _KEY_FEATURE_NAMES if k in feature_dict}
