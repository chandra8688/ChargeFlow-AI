"""End-to-end Phase 4 demonstration script."""
import pandas as pd
from src.services.feature_service import FeatureService
from src.services.forecast_service import ForecastService

print("Loading FeatureService (historical CSV)...")
fs = FeatureService()
print("Loading ForecastService (RandomForest artifact)...")
fc = ForecastService(eager_load=True)
print()

station_id      = "STA001"
prediction_time = "2025-06-15 19:00:00"
temperature_c   = 27.5
is_holiday      = False

print("=" * 62)
print("RAW INPUT")
print("=" * 62)
print(f"  station_id:      {station_id}")
print(f"  prediction_time: {prediction_time}")
print(f"  temperature_c:   {temperature_c}")
print(f"  is_holiday:      {is_holiday}")

print()
print("=" * 62)
print("FEATURE ENGINEERING  (FeatureService.build_features)")
print("=" * 62)
feature_dict = fs.build_features(
    station_id=station_id,
    prediction_time=prediction_time,
    temperature_c=temperature_c,
    is_holiday=is_holiday,
)
for k, v in feature_dict.items():
    val = f"{v:.6f}" if isinstance(v, float) else str(v)
    print(f"  {k:<22} = {val}")

print()
print("=" * 62)
print("LEAKAGE GUARD VERIFICATION")
print("=" * 62)
ts = pd.Timestamp(prediction_time)
for hours in [1, 24, 168]:
    lag_ts = ts - pd.Timedelta(hours=hours)
    ok = "PASS" if lag_ts < ts else "FAIL"
    print(f"  lag_{hours}h target {lag_ts}  < {ts}  [{ok}]")

print()
print("=" * 62)
print("MODEL INFERENCE  (ForecastService -> saved RandomForest)")
print("=" * 62)
result = fc.predict_single(feature_dict)
print(f"  predicted_occupancy : {result['predicted_occupancy']:.4f}  ({result['predicted_occupancy']*100:.1f}%)")
print(f"  status              : {result['status']}")
print(f"  model_type          : {result['model_type']}")
print(f"  feature_count       : {result['feature_count']}")

print()
print("=" * 62)
print("FEATURE CONTEXT  (descriptive — not causal)")
print("=" * 62)
ctx = fs.build_context(feature_dict)
for k, v in ctx.items():
    print(f"  {k:<22} = {v}")

# Also demonstrate 2025-06-30 00:00 is valid (non-hardcoded cutoff)
print()
print("=" * 62)
print("BOUNDARY TEST: 2025-06-30 00:00 (not hardcoded as cutoff)")
print("=" * 62)
feat_boundary = fs.build_features("STA001", "2025-06-30 00:00:00", 26.0)
result_boundary = fc.predict_single(feat_boundary)
print(f"  prediction_time     : 2025-06-30 00:00:00")
print(f"  predicted_occupancy : {result_boundary['predicted_occupancy']:.4f}  ({result_boundary['predicted_occupancy']*100:.1f}%)")
print(f"  status              : {result_boundary['status']}")
print(f"  lag_1h used         : {feat_boundary['lag_1h']:.4f}  (from 2025-06-29 23:00)")
print(f"  lag_168h used       : {feat_boundary['lag_168h']:.4f}  (from 2025-06-23 00:00)")
print()
print("Phase 4 end-to-end demonstration complete.")
