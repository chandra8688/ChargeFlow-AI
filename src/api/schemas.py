"""
ChargeFlow AI V2 — API Pydantic Schemas
==========================================
Defines the exact request and response contracts for the FastAPI endpoints.

Design choices:
  - Input schema mirrors the EXACT 16-feature contract from Phase 2 (DemandForecaster
    / ForecastService). Features are NOT changed here to suit the API; the API
    adapts to the model's contract.
  - Ranges are validated where they have a clear physical meaning:
      hour          : 0–23
      day_of_week   : 0–6
      month         : 1–12
      is_weekend    : 0 or 1
      is_holiday    : 0 or 1
      lag / rolling : 0.0–1.0 (they represent past occupancy_rate fractions)
      temperature_c : physically plausible for Indian metro cities (-5 to 50 C)
  - Cyclical sin/cos values must lie in [-1.0, 1.0].
  - Response exposes only fields that ForecastService actually returns — no
    fabricated confidence intervals or fabricated accuracy claims.
"""

from __future__ import annotations

import math
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Input: single prediction request ─────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    All 16 features required by the Phase 2 RandomForest model.
    Matches src/models/demand_forecaster.py::FEATURE_COLS exactly.
    """

    # Calendar
    hour:        int   = Field(..., ge=0,  le=23,  description="Hour of day (0–23)")
    day_of_week: int   = Field(..., ge=0,  le=6,   description="Day of week (0=Mon, 6=Sun)")
    month:       int   = Field(..., ge=1,  le=12,  description="Month of year (1–12)")

    # Cyclical encodings — sin/cos of hour and day
    hour_sin: float = Field(..., ge=-1.0, le=1.0, description="sin(2pi*hour/24)")
    hour_cos: float = Field(..., ge=-1.0, le=1.0, description="cos(2pi*hour/24)")
    day_sin:  float = Field(..., ge=-1.0, le=1.0, description="sin(2pi*day_of_week/7)")
    day_cos:  float = Field(..., ge=-1.0, le=1.0, description="cos(2pi*day_of_week/7)")

    # Boolean flags (0 or 1 encoded as int)
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, else 0")
    is_holiday: int = Field(..., ge=0, le=1, description="1 if public holiday, else 0")

    # Environmental
    temperature_c: float = Field(
        ..., ge=-5.0, le=50.0,
        description="Ambient temperature in Celsius (-5 to 50)"
    )

    # Station-grouped lag features (past occupancy_rate values → bounded [0,1])
    lag_1h:   float = Field(..., ge=0.0, le=1.0, description="Occupancy 1 hour ago")
    lag_24h:  float = Field(..., ge=0.0, le=1.0, description="Occupancy 24 hours ago (same hour yesterday)")
    lag_168h: float = Field(..., ge=0.0, le=1.0, description="Occupancy 168 hours ago (same hour last week)")

    # Station-grouped rolling statistics (also derived from past occupancy → [0,1])
    rolling_mean_6h:  float = Field(..., ge=0.0, le=1.0, description="Rolling 6h mean occupancy (lag-shifted)")
    rolling_mean_24h: float = Field(..., ge=0.0, le=1.0, description="Rolling 24h mean occupancy (lag-shifted)")
    rolling_std_24h:  float = Field(
        ..., ge=0.0, le=1.0,
        description="Rolling 24h std-dev of occupancy (lag-shifted)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "hour": 19, "day_of_week": 4, "month": 5,
            "hour_sin": -0.866, "hour_cos": 0.5,
            "day_sin": -0.782, "day_cos": 0.623,
            "is_weekend": 0, "is_holiday": 0,
            "temperature_c": 27.5,
            "lag_1h": 0.72, "lag_24h": 0.68, "lag_168h": 0.71,
            "rolling_mean_6h": 0.65, "rolling_mean_24h": 0.55, "rolling_std_24h": 0.08,
        }
    }}

    def to_feature_dict(self) -> dict:
        """Convert to a plain dict suitable for ForecastService.predict_single()."""
        return self.model_dump()


# ── Input: batch prediction request ──────────────────────────────────────────

MAX_BATCH_SIZE = 500  # prevent accidental very-large payloads

class BatchPredictionRequest(BaseModel):
    """Wraps a list of prediction requests for the /predict/batch endpoint."""
    items: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of prediction requests (1–{MAX_BATCH_SIZE} items)"
    )


# ── Output: single prediction response ───────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Response from POST /predict.
    Only exposes what ForecastService.predict_single() actually returns.
    No fabricated confidence intervals.
    """
    predicted_occupancy: float = Field(
        ..., description="Predicted occupancy_rate in [0.0, 1.0]"
    )
    status: str = Field(
        ..., description="Human-readable status: AVAILABLE | MODERATE | BUSY | CRITICAL"
    )
    model_type: str = Field(..., description="Model class used for inference")
    feature_count: int = Field(..., description="Number of features consumed by the model")


# ── Output: single item within a batch response ───────────────────────────────

class BatchPredictionItem(BaseModel):
    """One result within the batch response."""
    index: int = Field(..., description="Zero-based index corresponding to the input item")
    predicted_occupancy: float
    status: str
    model_type: str
    feature_count: int


class BatchPredictionResponse(BaseModel):
    """Response from POST /predict/batch."""
    count: int = Field(..., description="Number of predictions returned")
    results: List[BatchPredictionItem]


# ── Output: health check ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


# ── Output: model info ────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """
    Surfaces fields from demand_forecaster_metadata.json.
    Only fields present in the actual metadata are populated.
    """
    model_type:        str
    n_estimators:      Optional[int]   = None
    max_depth:         Optional[int]   = None
    min_samples_leaf:  Optional[int]   = None
    random_state:      Optional[int]   = None
    feature_count:     int
    feature_names:     List[str]
    target_col:        str
    trained_at:        Optional[str]   = None
    train_metrics:     Optional[dict]  = None
    val_metrics:       Optional[dict]  = None
    test_metrics:      Optional[dict]  = None


# ── Output: error detail ──────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Structured error response body returned on 4xx/5xx."""
    error: str
    detail: str


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Raw Prediction Schemas
# ══════════════════════════════════════════════════════════════════════════════
# These schemas support the POST /predict/raw endpoint.
# Callers supply human-understandable inputs; FeatureService derives the full
# 16-feature vector internally.

class RawPredictionRequest(BaseModel):
    """
    Minimal human-understandable request for /predict/raw.

    The backend (FeatureService) derives all 16 ML features from these 4 inputs
    plus the historical occupancy record for the specified station.
    """
    station_id:      str   = Field(
        ...,
        description="Station identifier (e.g. 'STA001'). Must exist in historical data.",
        examples=["STA001"],
    )
    prediction_time: str   = Field(
        ...,
        description=(
            "ISO-format datetime on an exact hour boundary, tz-naive. "
            "Sufficient history must exist for lag_1h, lag_24h, lag_168h, "
            "and 24-hour rolling windows. Example: '2025-06-15 19:00:00'."
        ),
        examples=["2025-06-15 19:00:00"],
    )
    temperature_c:   float = Field(
        ..., ge=-5.0, le=50.0,
        description="Ambient temperature in Celsius.",
        examples=[27.5],
    )
    is_holiday:      bool  = Field(
        False,
        description="True if the prediction date is a public holiday.",
        examples=[False],
    )


class RawPredictionResponse(BaseModel):
    """
    Response from POST /predict/raw.

    Includes the prediction, status, and a feature context block.
    The context block is DESCRIPTIVE only — it characterises the
    feature inputs, not a causal explanation of the model's decision.
    """
    station_id:           str
    prediction_time:      str
    predicted_occupancy:  float = Field(..., ge=0.0, le=1.0)
    status:               str
    model_type:           str
    feature_context:      dict  = Field(
        ...,
        description=(
            "Descriptive summary of the key feature values used in inference. "
            "This is feature characterisation, not a causal model explanation."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Explainability Schemas
# ══════════════════════════════════════════════════════════════════════════════

class FeatureImportanceItem(BaseModel):
    """Single feature importance entry (MDI from the trained RandomForest)."""
    feature:    str
    importance: float = Field(..., ge=0.0, le=1.0)


class FeatureImportanceResponse(BaseModel):
    """
    Response from GET /model/feature-importance.

    Reports global MDI (Mean Decrease in Impurity) importances from the
    saved RandomForest artifact.  All 16 features are included, sorted
    by importance descending.

    Terminology note:
        MDI importance measures predictive association in the training data.
        It does NOT measure causal effect.
    """
    model_type:  str
    n_features:  int
    importances: List[FeatureImportanceItem]


class FeatureContextItem(BaseModel):
    """
    One entry in the per-prediction feature context block.

    Combines a feature's global MDI rank with the value actually supplied
    for this specific prediction.  This is FEATURE CHARACTERISATION —
    it is NOT local attribution (no SHAP/LIME perturbation is used).
    """
    feature:    str
    importance: float  # global MDI score
    value:      float  # actual input value for this prediction


class TreeDispersionResult(BaseModel):
    """
    Per-prediction statistics across all individual RandomForest estimators.

    IMPORTANT — correct terminology:
        This is the "Tree Prediction Spread" or "Estimator Dispersion".
        It measures variation AMONG the 200 individual estimators for one input.
        It is NOT a calibrated confidence interval or uncertainty estimate.

    tree_mean is verified to be consistent with the aggregate model
    prediction returned by ForecastService.predict_single().
    """
    tree_mean:            float = Field(..., ge=0.0, le=1.0)
    tree_std:             float = Field(..., ge=0.0)
    tree_min:             float = Field(..., ge=0.0, le=1.0)
    tree_max:             float = Field(..., ge=0.0, le=1.0)
    p10:                  float = Field(..., ge=0.0, le=1.0)
    p90:                  float = Field(..., ge=0.0, le=1.0)
    estimator_count:      int
    status_consensus_pct: float = Field(..., ge=0.0, le=100.0)
    disclaimer:           str


class RawExplainResponse(BaseModel):
    """
    Response from POST /predict/raw/explain.

    Extends the raw prediction with:
      - top_feature_context: top-5 globally important features + actual values
      - tree_dispersion:     estimator spread statistics
      - prediction_id:       unique event ID (also logged to inference_log.jsonl)
      - inference_latency_ms: wall-clock time for the full pipeline
    """
    prediction_id:        str
    station_id:           str
    prediction_time:      str
    predicted_occupancy:  float = Field(..., ge=0.0, le=1.0)
    status:               str
    model_type:           str
    feature_context:      dict
    top_feature_context:  List[FeatureContextItem]
    tree_dispersion:      TreeDispersionResult
    inference_latency_ms: float


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6 — RAG Schemas
# ══════════════════════════════════════════════════════════════════════════════

class RAGQueryRequest(BaseModel):
    """
    Request model for POST /rag/query.
    """
    question: str = Field(..., min_length=2, description="User question about ChargeFlow system or EV domain.")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Maximum number of context chunks to retrieve.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for similarity threshold.")


class RAGSourceItem(BaseModel):
    """
    Metadata snippet for a retrieved context source chunk.
    """
    chunk_id: str
    source: str
    section_title: str
    text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class RAGQueryResponse(BaseModel):
    """
    Response model for POST /rag/query.
    """
    question: str
    answer: str
    grounded: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List[RAGSourceItem]
    llm_invoked: bool
    latency_ms: float


# ══════════════════════════════════════════════════════════════════════════════
# Phase 7 — AI Decision & Recommendation Schemas
# ══════════════════════════════════════════════════════════════════════════════

class DecisionRequest(BaseModel):
    """
    Request model for POST /recommend.
    """
    station_id: str = Field(..., description="Target station ID (e.g. STA001).")
    prediction_time: str = Field(..., description="Target timestamp in 'YYYY-MM-DD HH:MM:SS' format.")
    temperature_c: float = Field(28.0, ge=-20.0, le=60.0, description="Ambient temperature in Celsius.")
    is_holiday: bool = Field(False, description="Whether target day is a public holiday.")
    max_alternatives: Optional[int] = Field(3, ge=1, le=10, description="Maximum candidate alternatives to return.")
    include_rag_context: Optional[bool] = Field(False, description="Whether to attach optional RAG domain advice.")


class SelectedStationItem(BaseModel):
    """Target station prediction summary."""
    station_id: str
    name: str
    city: str
    charger_type: str
    predicted_occupancy: float = Field(..., ge=0.0, le=1.0)
    status: str


class AlternativeStationResponse(BaseModel):
    """Candidate alternative station ranking entry."""
    station_id: str
    name: str
    city: str
    charger_type: str
    predicted_occupancy: float = Field(..., ge=0.0, le=1.0)
    status: str
    distance_km: float = Field(..., ge=0.0)
    occupancy_delta: float
    occupancy_improvement: float


class DecisionPolicyItem(BaseModel):
    """Configurable decision policy threshold constants."""
    busy_threshold: float
    min_occupancy_improvement: float


class DecisionResponse(BaseModel):
    """
    Response model for POST /recommend.
    """
    selected_station: SelectedStationItem
    recommendation: str  # STAY | REROUTE | NO_BETTER_ALTERNATIVE
    recommendation_reason: str
    policy_thresholds: DecisionPolicyItem
    top_alternative: Optional[AlternativeStationResponse] = None
    alternatives: List[AlternativeStationResponse]
    tree_dispersion: Optional[TreeDispersionResult] = None
    top_feature_context: Optional[List[FeatureContextItem]] = None
    rag_context: Optional[dict] = None
    latency_ms: float

