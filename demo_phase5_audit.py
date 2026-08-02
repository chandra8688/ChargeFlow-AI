"""Phase 5 audit — probe feature importances and tree dispersion."""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from src.services.forecast_service import ForecastService
from src.services.feature_service import FeatureService

fs = FeatureService()
fc = ForecastService(eager_load=True)
forecaster = fc._forecaster
rf_model   = forecaster.model

# 1. Global feature importances
fi = forecaster.feature_importance()
print("== Global Feature Importances (MDI) ==")
for _, row in fi.iterrows():
    bar = "*" * int(row["importance"] * 200)
    print(f"  {row['feature']:<22} {row['importance']:.5f}  {bar}")
print()

# helper: call individual trees via numpy (no UserWarning)
def tree_preds_np(feat_dict, cols):
    x = np.array([[float(feat_dict[c]) for c in cols]])
    return np.clip(np.array([t.predict(x)[0] for t in rf_model.estimators_]), 0.0, 1.0)

# 2. Tree-level dispersion for evening peak
feat = fs.build_features("STA001", "2025-06-15 19:00:00", 27.5, False)
tp   = tree_preds_np(feat, forecaster.feature_cols)
print("== Tree Dispersion — Evening Peak (19:00) ==")
print(f"  final (mean of trees) : {tp.mean():.4f}  ({tp.mean()*100:.1f}%)")
print(f"  std across 200 trees  : {tp.std():.4f}")
print(f"  p10                   : {np.percentile(tp, 10):.4f}")
print(f"  p90                   : {np.percentile(tp, 90):.4f}")
print(f"  pct trees >= 0.90     : {(tp >= 0.90).mean()*100:.1f}%")
print()

# 3. Low-demand comparison
feat2 = fs.build_features("STA001", "2025-06-15 03:00:00", 22.0, False)
tp2   = tree_preds_np(feat2, forecaster.feature_cols)
print("== Tree Dispersion — 3 AM Off-peak ==")
print(f"  final (mean of trees) : {tp2.mean():.4f}  ({tp2.mean()*100:.1f}%)")
print(f"  std across 200 trees  : {tp2.std():.4f}")
print(f"  p10                   : {np.percentile(tp2, 10):.4f}")
print(f"  p90                   : {np.percentile(tp2, 90):.4f}")
print()

# 4. Zero-ablation group sensitivity
cols = forecaster.feature_cols
baseline = tp.mean()
groups = {
    "lag features":     ["lag_1h", "lag_24h", "lag_168h"],
    "rolling features": ["rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h"],
    "calendar":         ["hour", "day_of_week", "month"],
    "cyclical":         ["hour_sin", "hour_cos", "day_sin", "day_cos"],
    "flags":            ["is_weekend", "is_holiday"],
    "temperature":      ["temperature_c"],
}
print("== Feature Group Ablation Sensitivity (zero out group, measure delta) ==")
for gname, gcols in groups.items():
    ablated = dict(feat)
    for c in gcols:
        ablated[c] = 0.0
    ap = tree_preds_np(ablated, cols).mean()
    delta = ap - baseline
    print(f"  {gname:<20} baseline={baseline:.4f}  ablated={ap:.4f}  delta={delta:+.4f}")
