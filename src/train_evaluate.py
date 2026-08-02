"""
ChargeFlow AI V2 — Phase 2 Training & Evaluation Script
=========================================================
End-to-end pipeline:
  1. Generate time-series data from stations.csv
  2. Engineer features (Phase 1 preprocessor)
  3. Chronological train/val/test split
  4. Evaluate Seasonal Lag-24h Baseline on test set
  5. Train RandomForest DemandForecaster
  6. Evaluate model on test set
  7. Print baseline vs model comparison table
  8. Run error analysis (by city, hour, charger_type)
  9. Compute feature importance
 10. Save model + metadata to artifacts/models/
 11. Save all results to artifacts/evaluation_report.json

Usage:
    python src/train_evaluate.py

All metrics are computed from actual model execution — none are fabricated.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
DATA_PATH     = ARTIFACTS_DIR / "hourly_charging_data.csv"

# ── Import Phase 1 modules ────────────────────────────────────────────────────
from src.data.generate_timeseries import generate_hourly_timeseries
from src.data.preprocessor import DataPreprocessor

# ── Import Phase 2 modules ────────────────────────────────────────────────────
from src.models.demand_forecaster import (
    DemandForecaster, SeasonalBaseline, FEATURE_COLS, TARGET_COL
)


def _section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load or generate hourly dataset ───────────────────────────────
    _section("STEP 1 — DATA LOADING")
    if DATA_PATH.exists():
        print(f"Loading existing dataset from {DATA_PATH} ...")
        df_raw = pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found — generating now (180 days, seed=42)...")
        df_raw = generate_hourly_timeseries(num_days=180, seed=42)
        df_raw.to_csv(DATA_PATH, index=False)

    print(f"Raw dataset shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"Date range: {df_raw['timestamp'].min()} -> {df_raw['timestamp'].max()}")

    # ── STEP 2: Feature engineering ───────────────────────────────────────────
    _section("STEP 2 — FEATURE ENGINEERING")
    preprocessor = DataPreprocessor()
    df_feat = preprocessor.engineer_features(df_raw)
    print(f"Featured dataset: {df_feat.shape[0]:,} rows × {df_feat.shape[1]} columns")
    print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target column: {TARGET_COL}")
    print(f"Target range: [{df_feat[TARGET_COL].min():.4f}, {df_feat[TARGET_COL].max():.4f}]")

    # ── STEP 3: Chronological split ───────────────────────────────────────────
    _section("STEP 3 — CHRONOLOGICAL SPLIT (70 / 15 / 15)")
    train_df, val_df, test_df = preprocessor.split_chronological(df_feat)
    print(f"Train : {len(train_df):>8,} rows  "
          f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Val   : {len(val_df):>8,} rows  "
          f"{val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Test  : {len(test_df):>8,} rows  "
          f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Sanity check: no temporal overlap
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "OVERLAP: train/val!"
    assert val_df["timestamp"].max()   < test_df["timestamp"].min(), "OVERLAP: val/test!"
    print("\n[OK] No temporal overlap between splits.")

    # ── STEP 4: Seasonal Baseline ─────────────────────────────────────────────
    _section("STEP 4 — SEASONAL LAG-24H BASELINE (TEST SET)")
    baseline = SeasonalBaseline()
    baseline_test_metrics = baseline.evaluate(test_df)
    print(f"  Baseline — MAE : {baseline_test_metrics['MAE']:.5f}")
    print(f"  Baseline — RMSE: {baseline_test_metrics['RMSE']:.5f}")
    print(f"  Baseline — R²  : {baseline_test_metrics['R2']:.4f}")

    # ── STEP 5: Train RandomForest ────────────────────────────────────────────
    _section("STEP 5 — RANDOM FOREST TRAINING")
    forecaster = DemandForecaster(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=10,
        random_state=42,
    )
    t0 = time.perf_counter()
    train_results = forecaster.train(train_df, val_df)
    t1 = time.perf_counter()
    print(f"\nTraining time: {t1 - t0:.1f}s")

    # ── STEP 6: Test set evaluation ───────────────────────────────────────────
    _section("STEP 6 — FINAL TEST SET EVALUATION")
    test_metrics = forecaster.evaluate_test(test_df)

    # ── STEP 7: Comparison table ──────────────────────────────────────────────
    _section("STEP 7 — BASELINE vs ML MODEL COMPARISON")
    print(f"\n{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 56)
    print(f"{'Seasonal Lag-24h Baseline':<30} "
          f"{baseline_test_metrics['MAE']:>8.5f} "
          f"{baseline_test_metrics['RMSE']:>8.5f} "
          f"{baseline_test_metrics['R2']:>8.4f}")
    print(f"{'Random Forest (V2)':<30} "
          f"{test_metrics['MAE']:>8.5f} "
          f"{test_metrics['RMSE']:>8.5f} "
          f"{test_metrics['R2']:>8.4f}")
    mae_improve  = (baseline_test_metrics['MAE']  - test_metrics['MAE'])  / baseline_test_metrics['MAE'] * 100
    rmse_improve = (baseline_test_metrics['RMSE'] - test_metrics['RMSE']) / baseline_test_metrics['RMSE'] * 100
    print(f"\n  MAE  improvement over baseline: {mae_improve:+.1f}%")
    print(f"  RMSE improvement over baseline: {rmse_improve:+.1f}%")

    # ── STEP 8: Error analysis ────────────────────────────────────────────────
    _section("STEP 8 — ERROR ANALYSIS")
    error_results = forecaster.error_analysis(test_df)

    print("\n— By City —")
    print(error_results["by_city"].to_string(index=False))

    print("\n— By Charger Type —")
    print(error_results["by_charger_type"].to_string(index=False))

    print("\n— By Hour (top 5 worst, top 5 best MAE) —")
    by_hour = error_results["by_hour"].sort_values("MAE", ascending=False)
    print("Worst hours:")
    print(by_hour.head(5).to_string(index=False))
    print("Best hours:")
    print(by_hour.tail(5).to_string(index=False))

    # ── STEP 9: Feature importance ────────────────────────────────────────────
    _section("STEP 9 — FEATURE IMPORTANCE (MDI)")
    fi_df = forecaster.feature_importance()
    print(f"\n{'Feature':<22} {'Importance':>12}")
    print("-" * 36)
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"{row['feature']:<22} {row['importance']:>12.5f}  {bar}")

    fi_path = ARTIFACTS_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"\nFeature importance saved: {fi_path}")

    # ── STEP 10: Save model ───────────────────────────────────────────────────
    _section("STEP 10 — MODEL PERSISTENCE")
    paths = forecaster.save(
        model_dir=MODELS_DIR,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # ── STEP 11: Save evaluation report ──────────────────────────────────────
    _section("STEP 11 — SAVING EVALUATION REPORT")
    report = {
        "phase": "Phase 2 — Production ML Demand Forecasting",
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "dataset": {
            "raw_rows": int(df_raw.shape[0]),
            "featured_rows": int(df_feat.shape[0]),
            "train_rows": int(len(train_df)),
            "val_rows":   int(len(val_df)),
            "test_rows":  int(len(test_df)),
            "train_dates": {"min": str(train_df["timestamp"].min()), "max": str(train_df["timestamp"].max())},
            "val_dates":   {"min": str(val_df["timestamp"].min()),   "max": str(val_df["timestamp"].max())},
            "test_dates":  {"min": str(test_df["timestamp"].min()),  "max": str(test_df["timestamp"].max())},
        },
        "baseline": {
            "name": "Seasonal Lag-24h Baseline",
            "rule": "prediction(t) = occupancy_rate(t-24h)",
            "test_metrics": baseline_test_metrics,
        },
        "random_forest": {
            "n_estimators": forecaster.n_estimators,
            "max_depth": forecaster.max_depth,
            "min_samples_leaf": forecaster.min_samples_leaf,
            "random_state": forecaster.random_state,
            "train_metrics": train_results["train"],
            "val_metrics":   train_results["val"],
            "test_metrics":  test_metrics,
        },
        "improvement_over_baseline_pct": {
            "MAE":  round(mae_improve, 2),
            "RMSE": round(rmse_improve, 2),
        },
        "error_analysis": {
            "by_city":         error_results["by_city"].to_dict(orient="records"),
            "by_charger_type": error_results["by_charger_type"].to_dict(orient="records"),
            "by_hour_worst5":  error_results["by_hour"].sort_values("MAE", ascending=False).head(5).to_dict(orient="records"),
            "by_hour_best5":   error_results["by_hour"].sort_values("MAE", ascending=False).tail(5).to_dict(orient="records"),
        },
        "feature_importance_top5": fi_df.head(5).to_dict(orient="records"),
        "model_paths": paths,
    }

    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved: {report_path}")

    _section("PHASE 2 COMPLETE")
    print(f"  Model    : {paths['model_path']}")
    print(f"  Metadata : {paths['metadata_path']}")
    print(f"  Report   : {report_path}")
    print(f"\n  Test MAE  = {test_metrics['MAE']:.5f}")
    print(f"  Test RMSE = {test_metrics['RMSE']:.5f}")
    print(f"  Test R²   = {test_metrics['R2']:.4f}")
    print(f"\n  Baseline MAE  = {baseline_test_metrics['MAE']:.5f}")
    print(f"  MAE Improvement: {mae_improve:+.1f}%")


if __name__ == "__main__":
    main()
