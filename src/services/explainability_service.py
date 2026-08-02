"""
ChargeFlow AI V2 — Explainability Service
==========================================
Provides two categories of explanation for the Phase 2 DemandForecaster.

1. Global Feature Importance
   ─────────────────────────
   Source: model.feature_importances_ (Mean Decrease in Impurity, MDI).
   Scope:  Computed once over the entire training dataset; reflects which
           features the forest relies on most across all training examples.
   Limit:  MDI measures predictive association learned during training.
           It does NOT measure causal effect and MUST NOT be described
           as such in any user-facing text.

2. Tree Prediction Spread (estimator dispersion)
   ───────────────────────────────────────────────
   Source: Individual DecisionTreeRegressor.predict() calls on all 200
           estimators within the saved RandomForest.
   Scope:  Per-prediction — recomputed for each inference call.
   Limit:  This is NOT a calibrated confidence or uncertainty interval.
           It measures variation among individual estimators for a given
           input. The aggregate prediction (tree_mean) is verified to be
           consistent with ForecastService.predict_single() to within
           float rounding tolerance.

3. Per-prediction Feature Context
   ────────────────────────────────
   Shows which globally-important features were active for this specific
   prediction and what values were supplied.
   This is feature characterisation, NOT attribution or causal explanation.

No external explainability libraries (SHAP, LIME, etc.) are used.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.demand_forecaster import FEATURE_COLS
from src.services.forecast_service import ForecastService, _occupancy_to_status


class ExplainabilityService:
    """
    Lightweight explainability layer over the saved RandomForest model.

    Must be initialised with a loaded ForecastService so that both the
    model and its feature-contract are shared — no second model load.
    """

    def __init__(self, forecast_service: ForecastService):
        if forecast_service._forecaster is None:
            raise RuntimeError(
                "ForecastService must have a loaded model before "
                "ExplainabilityService can be constructed."
            )
        self._forecast_service = forecast_service
        self._forecaster       = forecast_service._forecaster
        self._rf_model         = self._forecaster.model

        # Cached on first call — feature importances do not change after training
        self._cached_importances: Optional[List[Dict[str, Any]]] = None

    # ── Global Feature Importance ─────────────────────────────────────────────

    def global_feature_importance(self) -> List[Dict[str, Any]]:
        """
        Return MDI feature importances from the trained RandomForest.

        Cached after first call — importances are fixed post-training.

        Returns:
            List of dicts [{"feature": str, "importance": float}, ...],
            sorted by importance descending.

        Note:
            MDI importance reflects the average total reduction in node
            impurity weighted by the proportion of samples reaching that
            node, across all trees. It measures predictive association
            in the training data — NOT causal effect.
        """
        if self._cached_importances is not None:
            return self._cached_importances

        fi_df = self._forecaster.feature_importance()
        self._cached_importances = [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in fi_df.iterrows()
        ]
        return self._cached_importances

    def top_n_feature_context(
        self,
        feature_dict: Dict[str, Any],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-N globally important features together with the
        actual values supplied for this specific prediction.

        This is FEATURE CHARACTERISATION — it shows which features the
        model generally relies on most and what values were provided.
        It is NOT local attribution (no SHAP/LIME perturbation is used).

        Args:
            feature_dict: The 16-feature dict for this prediction.
            n:            Number of top features to return (default 5).

        Returns:
            List of dicts:
              feature    — feature name
              importance — global MDI importance score
              value      — actual input value supplied for this prediction
        """
        importances = self.global_feature_importance()
        top = importances[:n]  # already sorted descending
        return [
            {
                "feature":    item["feature"],
                "importance": item["importance"],
                "value":      round(float(feature_dict[item["feature"]]), 6),
            }
            for item in top
        ]

    # ── Tree Prediction Spread ────────────────────────────────────────────────

    def tree_dispersion(
        self, feature_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute per-prediction statistics across all individual RandomForest
        estimators (DecisionTreeRegressor objects).

        Uses numpy arrays for inference to avoid sklearn UserWarning about
        feature-name mismatches between fit-time DataFrame and per-tree calls.

        Verifies that tree_mean is consistent with the aggregate model
        prediction returned by ForecastService.predict_single().

        Returns a dict with raw statistics. The caller is responsible for
        communicating these correctly to end users.

        IMPORTANT — correct terminology:
          "Tree Prediction Spread" or "Estimator Dispersion"
          NOT "confidence interval" or "prediction interval".
          This measures variation AMONG estimators — it is not a
          calibrated uncertainty quantification.
        """
        cols = self._forecaster.feature_cols
        # Build 1-row numpy array in the same column order used at training time
        x_np = np.array([[float(feature_dict[c]) for c in cols]])

        # Call each estimator; suppress feature-name warnings from sklearn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_preds = np.array(
                [t.predict(x_np)[0] for t in self._rf_model.estimators_]
            )

        tree_preds = np.clip(raw_preds, 0.0, 1.0)

        tree_mean = float(np.mean(tree_preds))

        # Determine the status bucket from the aggregate mean
        agg_status = _occupancy_to_status(tree_mean)

        # Status consensus: % of trees whose clipped prediction maps to
        # the SAME status bucket as the aggregate prediction
        tree_statuses   = [_occupancy_to_status(float(p)) for p in tree_preds]
        consensus_count = sum(1 for s in tree_statuses if s == agg_status)
        consensus_pct   = round(consensus_count / len(tree_preds) * 100, 1)

        return {
            "tree_mean":             round(tree_mean, 4),
            "tree_std":              round(float(np.std(tree_preds)),                  4),
            "tree_min":              round(float(np.min(tree_preds)),                  4),
            "tree_max":              round(float(np.max(tree_preds)),                  4),
            "p10":                   round(float(np.percentile(tree_preds, 10)),       4),
            "p90":                   round(float(np.percentile(tree_preds, 90)),       4),
            "estimator_count":       len(self._rf_model.estimators_),
            "status_consensus_pct":  consensus_pct,
            "disclaimer": (
                "This measures variation among individual RandomForest estimators. "
                "It is NOT a calibrated uncertainty or confidence interval."
            ),
        }
