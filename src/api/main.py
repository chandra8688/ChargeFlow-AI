"""
ChargeFlow AI V2 — FastAPI Demand Forecast API
================================================
Serves the Phase 2 RandomForest demand forecasting model over HTTP.

Architecture:
    HTTP client
        -> FastAPI (this file)
            -> ForecastService  (src/services/forecast_service.py)
                -> DemandForecaster (src/models/demand_forecaster.py)
                    -> artifacts/models/demand_forecaster.joblib

Key design decisions:
  - ForecastService is instantiated ONCE at application startup via FastAPI
    lifespan and stored in app.state.  This avoids reloading the 107 MB
    RandomForest on every request.
  - The API layer contains NO inference logic — it only translates between
    HTTP and ForecastService.predict_single() / predict_batch().
  - Pydantic schemas enforce the exact 16-feature contract from Phase 2.
  - Model-unavailable errors surface as HTTP 503 (not 500) so clients can
    distinguish infrastructure problems from application bugs.

To run locally:
    python -m uvicorn src.api.main:app --reload

Swagger UI: http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.services.forecast_service import ForecastService
from src.services.feature_service import (
    FeatureService,
    UnknownStationError,
    InsufficientHistoryError,
)
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
    HealthResponse,
    ModelInfoResponse,
    ErrorDetail,
    MAX_BATCH_SIZE,
    RawPredictionRequest,
    RawPredictionResponse,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    FeatureContextItem,
    TreeDispersionResult,
    RawExplainResponse,
    RAGQueryRequest,
    RAGSourceItem,
    RAGQueryResponse,
    DecisionRequest,
    SelectedStationItem,
    AlternativeStationResponse,
    DecisionPolicyItem,
    DecisionResponse,
)
from src.services.explainability_service import ExplainabilityService
from src.services.inference_logger import InferenceLogger
from src.services.decision_service import DecisionService
from src.rag.rag_service import RAGService
from src.rag.llm_provider import GeminiLLMProvider

logger = logging.getLogger("chargeflow.api")


# ── Application Lifespan ──────────────────────────────────────────────────────
# FastAPI lifespan replaces the deprecated @app.on_event("startup") pattern.
# The model is loaded once here and stored in app.state so that every request
# handler can access it via request.app.state.forecast_service without
# reloading the artifact from disk.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ForecastService and FeatureService on startup; clean up on shutdown."""
    logger.info("ChargeFlow API starting — loading ForecastService ...")
    try:
        app.state.forecast_service = ForecastService(eager_load=True)
        app.state.model_loaded = True
        logger.info("ForecastService loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load ForecastService: %s", exc)
        app.state.forecast_service = None
        app.state.model_loaded = False

    # FeatureService loads the 18 MB historical CSV once at startup.
    # All /predict/raw requests reuse this in-memory structure.
    logger.info("ChargeFlow API — loading FeatureService (historical CSV) ...")
    try:
        app.state.feature_service = FeatureService()
        app.state.feature_service_loaded = True
        logger.info("FeatureService loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load FeatureService: %s", exc)
        app.state.feature_service = None
        app.state.feature_service_loaded = False

    # ExplainabilityService wraps the same ForecastService — no second model load.
    logger.info("ChargeFlow API — loading ExplainabilityService ...")
    try:
        if app.state.forecast_service is not None:
            app.state.explainability_service = ExplainabilityService(
                app.state.forecast_service
            )
            logger.info("ExplainabilityService loaded successfully.")
        else:
            app.state.explainability_service = None
            logger.warning("Skipping ExplainabilityService: ForecastService unavailable.")
    except Exception as exc:
        logger.error("Failed to load ExplainabilityService: %s", exc)
        app.state.explainability_service = None

    # InferenceLogger is lightweight (no model loading) — always available.
    logger.info("ChargeFlow API — initialising InferenceLogger ...")
    try:
        app.state.inference_logger = InferenceLogger()
        logger.info("InferenceLogger initialised at: %s",
                    app.state.inference_logger.log_path())
    except Exception as exc:
        logger.error("Failed to initialise InferenceLogger: %s", exc)
        app.state.inference_logger = None

    # RAGService loads repository knowledge chunks into vector index on startup.
    logger.info("ChargeFlow API — loading RAGService (Knowledge Base) ...")
    try:
        app.state.rag_service = RAGService(llm_provider=GeminiLLMProvider())
        num_chunks = app.state.rag_service.initialize()
        logger.info("RAGService initialized successfully with %d chunks.", num_chunks)
    except Exception as exc:
        logger.error("Failed to initialize RAGService: %s", exc)
        app.state.rag_service = None

    # DecisionService orchestrates real ML forecasts, candidate selection & ranking.
    logger.info("ChargeFlow API — loading DecisionService ...")
    try:
        if app.state.forecast_service and app.state.feature_service:
            app.state.decision_service = DecisionService(
                forecast_service=app.state.forecast_service,
                feature_service=app.state.feature_service,
                explainability_service=app.state.explainability_service,
                rag_service=app.state.rag_service,
            )
            logger.info("DecisionService initialized successfully.")
        else:
            app.state.decision_service = None
            logger.warning("Skipping DecisionService: Required forecast/feature services unavailable.")
    except Exception as exc:
        logger.error("Failed to initialize DecisionService: %s", exc)
        app.state.decision_service = None

    yield  # application runs between here and the next line

    logger.info("ChargeFlow API shutting down.")


# ── FastAPI Instance ──────────────────────────────────────────────────────────

app = FastAPI(
    title="ChargeFlow AI V2 — Demand Forecast API",
    description=(
        "Production-style inference API for EV charging station demand forecasting. "
        "Built on a RandomForest model trained on 180 days of hourly Indian EV station data. "
        "Phase 3 of the ChargeFlow AI V2 portfolio project."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Global Exception Handler ──────────────────────────────────────────────────
# Catches any unhandled exception and returns a clean JSON 500 instead of
# exposing the raw Python traceback to API clients.

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Dependency: get ForecastService ──────────────────────────────────────────
# A simple helper that reads from app.state and raises 503 if the model
# failed to load at startup.  All prediction endpoints use this.

def _get_service(request: Request) -> ForecastService:
    """
    Return the ForecastService from app.state.
    Raises HTTP 503 if the model was not loaded at startup.
    """
    service: ForecastService | None = getattr(request.app.state, "forecast_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is unavailable. The demand forecaster artifact could not "
                "be loaded at startup. Run 'python -m src.train_evaluate' first."
            ),
        )
    return service


def _get_feature_service(request: Request) -> FeatureService:
    """
    Return the FeatureService from app.state.
    Raises HTTP 503 if the historical CSV failed to load at startup.
    """
    fs: FeatureService | None = getattr(request.app.state, "feature_service", None)
    if fs is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "FeatureService is unavailable. The historical occupancy CSV could not "
                "be loaded at startup. Ensure artifacts/hourly_charging_data.csv exists."
            ),
        )
    return fs


# ── Endpoint: GET /health ─────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Infrastructure"],
)
async def health(request: Request):
    """
    Returns service health and whether the ML model is loaded.
    `model_loaded` reflects actual runtime state — not a hard-coded value.
    """
    return HealthResponse(
        status="healthy",
        service="ChargeFlow AI Demand Forecast API",
        # getattr is used defensively: if lifespan hasn't set the attribute yet
        # (e.g. during testing without context manager), return False rather than
        # raising an AttributeError that triggers the 500 handler.
        model_loaded=getattr(request.app.state, "model_loaded", False),
    )


# ── Endpoint: GET /model/info ─────────────────────────────────────────────────

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["Model"],
)
async def model_info(request: Request):
    """
    Returns metadata about the loaded model — type, hyperparameters,
    feature list, and training/validation/test metrics.

    All values come directly from the model metadata JSON artifact;
    nothing is fabricated.
    """
    service = _get_service(request)
    meta = service.model_metadata

    if not meta:
        raise HTTPException(
            status_code=503,
            detail="Model metadata is unavailable.",
        )

    return ModelInfoResponse(
        model_type=meta.get("model_type", "Unknown"),
        n_estimators=meta.get("n_estimators"),
        max_depth=meta.get("max_depth"),
        min_samples_leaf=meta.get("min_samples_leaf"),
        random_state=meta.get("random_state"),
        feature_count=meta.get("n_features", len(meta.get("feature_cols", []))),
        feature_names=meta.get("feature_cols", []),
        target_col=meta.get("target_col", "occupancy_rate"),
        trained_at=meta.get("trained_at"),
        train_metrics=meta.get("train_metrics"),
        val_metrics=meta.get("val_metrics"),
        test_metrics=meta.get("test_metrics"),
    )


# ── Endpoint: POST /predict ───────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single-station occupancy prediction",
    tags=["Inference"],
)
async def predict(body: PredictionRequest, request: Request):
    """
    Predict the occupancy_rate for a single EV charging station at a given hour.

    Accepts all 16 engineered features required by the Phase 2 model.
    Returns predicted occupancy in [0.0, 1.0] plus a human-readable status label.

    **Note:** This endpoint does NOT perform feature engineering.
    Callers must supply pre-engineered features (lags, rolling stats, cyclical encodings).
    These are the same features computed by `src/data/preprocessor.py`.
    """
    service = _get_service(request)
    try:
        result = service.predict_single(body.to_feature_dict())
    except ValueError as exc:
        # ForecastService raises ValueError for invalid input (e.g. lag out of range)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error on /predict")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictionResponse(**result)


# ── Endpoint: POST /predict/batch ─────────────────────────────────────────────

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch occupancy prediction",
    tags=["Inference"],
)
async def predict_batch(body: BatchPredictionRequest, request: Request):
    """
    Predict occupancy_rate for up to {MAX_BATCH_SIZE} station-hour observations
    in one request.

    Pydantic validates every item in the batch before inference begins.
    If any item is schema-invalid, the entire request is rejected with HTTP 422
    before reaching the model — no partial results with silent failures.

    Returns results in input order, each tagged with its zero-based index.
    """
    service = _get_service(request)

    # Convert list of PredictionRequest objects to a DataFrame the service expects
    records = [item.to_feature_dict() for item in body.items]
    df_input = pd.DataFrame(records)

    try:
        df_result = service.predict_batch(df_input)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference error on /predict/batch")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")

    results = [
        BatchPredictionItem(
            index=i,
            predicted_occupancy=float(row["predicted_occupancy"]),
            status=str(row["predicted_status"]),
            model_type="RandomForestRegressor",
            feature_count=len(service.feature_cols),
        )
        for i, row in df_result.iterrows()
    ]

    return BatchPredictionResponse(count=len(results), results=results)


# ── Endpoint: POST /predict/raw ───────────────────────────────────────────────

@app.post(
    "/predict/raw",
    response_model=RawPredictionResponse,
    summary="Raw-input occupancy prediction (with automatic feature engineering)",
    tags=["Inference"],
    responses={
        404: {"description": "Unknown station_id"},
        422: {"description": "Malformed timestamp, temp out of range, or insufficient history"},
        503: {"description": "Model or FeatureService unavailable"},
    },
)
async def predict_raw(body: RawPredictionRequest, request: Request):
    """
    End-to-end occupancy prediction from human-understandable inputs.

    Architecture::

        POST /predict/raw
        {station_id, prediction_time, temperature_c, is_holiday}
                ↓
        FeatureService.build_features()     ← derives all 16 ML features
                ↓                            from historical occupancy data
        ForecastService.predict_single()    ← calls saved RandomForest
                ↓
        RawPredictionResponse               ← occupancy + status + context

    The caller does NOT need to compute any ML features manually.
    FeatureService enforces strict leakage prevention: every historical
    lookup uses only observations with timestamp < prediction_time.

    **Constraints on prediction_time:**
    - Must be an exact hour boundary (e.g. '2025-06-15 19:00:00').
    - Sufficient historical observations must exist for lag_1h, lag_24h,
      lag_168h, rolling_mean_6h, rolling_mean_24h, rolling_std_24h.
      The service checks availability dynamically — no hardcoded cutoff.
    """
    forecast_svc = _get_service(request)
    feature_svc  = _get_feature_service(request)

    # Step 1 — Feature engineering (raises 404/422 on invalid input)
    try:
        feature_dict = feature_svc.build_features(
            station_id=body.station_id,
            prediction_time=body.prediction_time,
            temperature_c=body.temperature_c,
            is_holiday=body.is_holiday,
        )
    except UnknownStationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("FeatureService error on /predict/raw")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}")

    # Step 2 — Model inference (ForecastService — unchanged from Phase 3)
    try:
        result = forecast_svc.predict_single(feature_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("ForecastService error on /predict/raw")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # Step 3 — Build descriptive feature context (not a causal explanation)
    context = feature_svc.build_context(feature_dict)

    return RawPredictionResponse(
        station_id=body.station_id,
        prediction_time=body.prediction_time,
        predicted_occupancy=float(result["predicted_occupancy"]),
        status=str(result["status"]),
        model_type=str(result["model_type"]),
        feature_context=context,
    )


# ── Phase 5 Helpers ────────────────────────────────────────────────────────

def _get_explainability_service(request: Request) -> ExplainabilityService:
    svc: ExplainabilityService | None = getattr(
        request.app.state, "explainability_service", None
    )
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="ExplainabilityService is unavailable (model artifact not loaded).",
        )
    return svc


def _get_inference_logger(request: Request) -> InferenceLogger | None:
    """Return InferenceLogger from app.state; returns None (not 503) on failure."""
    return getattr(request.app.state, "inference_logger", None)


# ── Endpoint: GET /model/feature-importance ───────────────────────────────────

@app.get(
    "/model/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Global feature importance (MDI) from the trained RandomForest",
    tags=["Model"],
    responses={503: {"description": "Model artifact not loaded"}},
)
async def feature_importance(request: Request):
    """
    Return Mean Decrease in Impurity (MDI) feature importances from the
    saved RandomForest artifact.  All 16 features are included, sorted
    by importance descending.

    Terminology note:
        MDI importance measures predictive association learned during training.
        It does NOT measure causal effect on occupancy.
        Importances sum to approximately 1.0 across all features.
    """
    expl_svc = _get_explainability_service(request)
    service  = _get_service(request)
    meta     = service.model_metadata or {}

    raw = expl_svc.global_feature_importance()
    return FeatureImportanceResponse(
        model_type=meta.get("model_type", "RandomForestRegressor"),
        n_features=len(raw),
        importances=[FeatureImportanceItem(**item) for item in raw],
    )


# ── Endpoint: POST /predict/raw/explain ────────────────────────────────────

@app.post(
    "/predict/raw/explain",
    response_model=RawExplainResponse,
    summary="Raw-input prediction with feature context and tree prediction spread",
    tags=["Inference"],
    responses={
        404: {"description": "Unknown station_id"},
        422: {"description": "Malformed input or insufficient history"},
        503: {"description": "Model or services unavailable"},
    },
)
async def predict_raw_explain(body: RawPredictionRequest, request: Request):
    """
    End-to-end prediction with explainability output.

    Extends POST /predict/raw with:
      top_feature_context  — top-5 globally important features + their actual
                              input values for this prediction.
                              This is FEATURE CHARACTERISATION, not local attribution.
      tree_dispersion      — estimator spread statistics across all 200 RF trees.
                              Labelled "Tree Prediction Spread".
                              This is NOT a calibrated confidence interval.
      prediction_id        — uuid4; also written to logs/inference_log.jsonl.
      inference_latency_ms — wall-clock time for feature engineering + inference.

    The existing POST /predict/raw endpoint contract is unchanged.
    """
    import time

    forecast_svc  = _get_service(request)
    feature_svc   = _get_feature_service(request)
    expl_svc      = _get_explainability_service(request)
    inf_logger    = _get_inference_logger(request)

    # ── Step 1: Feature engineering (identical to /predict/raw) ───────────
    t_start = time.perf_counter()
    try:
        feature_dict = feature_svc.build_features(
            station_id=body.station_id,
            prediction_time=body.prediction_time,
            temperature_c=body.temperature_c,
            is_holiday=body.is_holiday,
        )
    except UnknownStationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("FeatureService error on /predict/raw/explain")
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {exc}")

    # ── Step 2: Aggregate model inference (identical to /predict/raw) ─────
    try:
        result = forecast_svc.predict_single(feature_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("ForecastService error on /predict/raw/explain")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # ── Step 3: Explainability ──────────────────────────────────────────
    top_ctx     = expl_svc.top_n_feature_context(feature_dict, n=5)
    dispersion  = expl_svc.tree_dispersion(feature_dict)
    feat_ctx    = feature_svc.build_context(feature_dict)

    latency_ms  = (time.perf_counter() - t_start) * 1000

    # ── Step 4: Inference logging ───────────────────────────────────────
    meta = forecast_svc.model_metadata or {}
    model_version = meta.get("trained_at", "unknown")

    prediction_id = "unknown"
    if inf_logger is not None:
        try:
            prediction_id = inf_logger.log(
                station_id=body.station_id,
                prediction_time=body.prediction_time,
                predicted_occupancy=float(result["predicted_occupancy"]),
                status=str(result["status"]),
                model_version=model_version,
                inference_latency_ms=latency_ms,
                source="api",
                key_features=InferenceLogger.extract_key_features(feature_dict),
            )
        except Exception as log_exc:
            logger.warning("InferenceLogger failed (non-fatal): %s", log_exc)

    return RawExplainResponse(
        prediction_id=prediction_id,
        station_id=body.station_id,
        prediction_time=body.prediction_time,
        predicted_occupancy=float(result["predicted_occupancy"]),
        status=str(result["status"]),
        model_type=str(result["model_type"]),
        feature_context=feat_ctx,
        top_feature_context=[FeatureContextItem(**item) for item in top_ctx],
        tree_dispersion=TreeDispersionResult(**dispersion),
        inference_latency_ms=round(latency_ms, 2),
    )


# ── Phase 6 Helper ────────────────────────────────────────────────────────

def _get_rag_service(request: Request) -> RAGService:
    svc: RAGService | None = getattr(request.app.state, "rag_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="RAGService is unavailable (knowledge base failed to initialize).",
        )
    return svc


# ── Endpoint: POST /rag/query ───────────────────────────────────────────────

@app.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    summary="Knowledge Intelligence query (RAG over ChargeFlow documents)",
    tags=["Knowledge Assistant"],
    responses={
        422: {"description": "Malformed query input"},
        503: {"description": "RAG Service unavailable"},
    },
)
async def rag_query(body: RAGQueryRequest, request: Request):
    """
    Knowledge Intelligence query endpoint.

    Decoupled from ML prediction endpoints (/predict, /predict/raw).
    Searches ChargeFlow AI project documentation and metadata artifacts,
    evaluates similarity score against threshold, and generates a grounded response.

    If evidence is below threshold, returns grounded=False and refusal answer
    WITHOUT invoking LLM generation.
    """
    rag_svc = _get_rag_service(request)
    try:
        res = rag_svc.query(
            question=body.question,
            top_k=body.top_k or 3,
            threshold=body.threshold,
        )
        return RAGQueryResponse(
            question=res["question"],
            answer=res["answer"],
            grounded=res["grounded"],
            confidence_score=res["confidence_score"],
            sources=[RAGSourceItem(**src) for src in res["sources"]],
            llm_invoked=res["llm_invoked"],
            latency_ms=res["latency_ms"],
        )
    except Exception as exc:
        logger.exception("RAG error on /rag/query")
        raise HTTPException(status_code=500, detail=f"RAG query processing failed: {exc}")


# ── Phase 7 Helper ────────────────────────────────────────────────────────

def _get_decision_service(request: Request) -> DecisionService:
    svc: DecisionService | None = getattr(request.app.state, "decision_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="DecisionService is unavailable (required ML services failed to initialize).",
        )
    return svc


# ── Endpoint: POST /recommend ───────────────────────────────────────────────

@app.post(
    "/recommend",
    response_model=DecisionResponse,
    summary="AI Decision Engine rerouting & candidate recommendation",
    tags=["Decision Engine"],
    responses={
        404: {"description": "Unknown station_id"},
        422: {"description": "Malformed input or insufficient history"},
        503: {"description": "Decision Engine unavailable"},
    },
)
async def recommend(body: DecisionRequest, request: Request):
    """
    AI Decision Engine endpoint.

    Evaluates target station predicted occupancy using the saved RandomForest model.
    Selects candidate alternative stations in the same city matching charger standards,
    predicts candidate demand at the exact same timestamp, ranks candidates deterministically
    (predicted_occupancy ASC, distance_km ASC), and applies transparent decision policy.
    """
    dec_svc = _get_decision_service(request)
    try:
        res = dec_svc.recommend(
            station_id=body.station_id,
            prediction_time=body.prediction_time,
            temperature_c=body.temperature_c,
            is_holiday=body.is_holiday,
            max_alternatives=body.max_alternatives or 3,
            include_rag_context=body.include_rag_context or False,
        )

        top_alt = (
            AlternativeStationResponse(**res["top_alternative"])
            if res["top_alternative"]
            else None
        )
        alts = [AlternativeStationResponse(**item) for item in res["alternatives"]]

        tree_disp = (
            TreeDispersionResult(**res["tree_dispersion"])
            if res["tree_dispersion"]
            else None
        )

        top_ctx = (
            [FeatureContextItem(**item) for item in res["top_feature_context"]]
            if res["top_feature_context"]
            else None
        )

        return DecisionResponse(
            selected_station=SelectedStationItem(**res["selected_station"]),
            recommendation=res["recommendation"],
            recommendation_reason=res["recommendation_reason"],
            policy_thresholds=DecisionPolicyItem(**res["policy_thresholds"]),
            top_alternative=top_alt,
            alternatives=alts,
            tree_dispersion=tree_disp,
            top_feature_context=top_ctx,
            rag_context=res["rag_context"],
            latency_ms=res["latency_ms"],
        )
    except UnknownStationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InsufficientHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("DecisionService error on /recommend")
        raise HTTPException(status_code=500, detail=f"Decision engine failed: {exc}")

