"""
ChargeFlow AI V2 — Data Preprocessor & Feature Engineering
============================================================
Performs time-series feature engineering, strict chronological train/val/test
splitting, and rigorous validation checks for EV demand forecasting:

1. Target Definition: occupancy_rate = occupied_slots / total_slots
2. Calendar Features: hour, day_of_week, month
3. Cyclical Features: hour_sin, hour_cos, day_sin, day_cos
4. Station-grouped Lags: lag_1h, lag_24h, lag_168h
5. Station-grouped Rolling Stats: rolling_mean_6h, rolling_mean_24h, rolling_std_24h
6. Zero Target Leakage: All lag & rolling features use shift(1) prior to computation
7. Chronological Split: Train (70%), Validation (15%), Test (15%)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class DataPreprocessor:
    """
    Preprocessor for station-level hourly EV charging demand time-series.
    """

    FEATURE_COLS = [
        "hour", "day_of_week", "month",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "is_weekend", "is_holiday", "temperature_c",
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h"
    ]
    TARGET_COL = "occupancy_rate"

    def __init__(self):
        pass

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers time-series features while strictly avoiding target leakage.

        Args:
            df: Raw hourly time-series DataFrame from generate_timeseries.py.

        Returns:
            Preprocessed DataFrame with lag and rolling features, NaNs dropped.
        """
        df = df.copy()
        
        # Ensure timestamp is datetime and sorted
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)

        # 1. Calendar Features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        # 2. Cyclical Encodings (Sine / Cosine Transformation)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        # Convert booleans to int (0 / 1)
        df["is_weekend"] = df["is_weekend"].astype(int)
        df["is_holiday"] = df["is_holiday"].astype(int)

        # 3. Station-Grouped Lag Features (Strict Historical Shift)
        df["lag_1h"]   = df.groupby("station_id")[self.TARGET_COL].shift(1)
        df["lag_24h"]  = df.groupby("station_id")[self.TARGET_COL].shift(24)
        df["lag_168h"] = df.groupby("station_id")[self.TARGET_COL].shift(168)

        # 4. Station-Grouped Rolling Features (Shift by 1 BEFORE rolling to prevent leakage)
        hist_series = df.groupby("station_id")[self.TARGET_COL].shift(1)
        
        df["rolling_mean_6h"]  = hist_series.groupby(df["station_id"]).transform(lambda s: s.rolling(6).mean())
        df["rolling_mean_24h"] = hist_series.groupby(df["station_id"]).transform(lambda s: s.rolling(24).mean())
        df["rolling_std_24h"]  = hist_series.groupby(df["station_id"]).transform(lambda s: s.rolling(24).std())

        # Drop NaNs resulting from max lag (168h) and initial rolling windows
        df_clean = df.dropna().reset_index(drop=True)
        return df_clean

    def split_chronological(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronologically splits the DataFrame into Train, Validation, and Test sets.

        Args:
            df: Preprocessed DataFrame with features.
            train_ratio: Ratio for training set (default 0.70).
            val_ratio: Ratio for validation set (default 0.15).
            test_ratio: Ratio for test set (default 0.15).

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

        # Unique sorted timestamps across the dataset
        unique_timestamps = np.sort(df["timestamp"].unique())
        n_timestamps = len(unique_timestamps)

        train_cutoff_idx = int(n_timestamps * train_ratio)
        val_cutoff_idx   = int(n_timestamps * (train_ratio + val_ratio))

        train_cutoff_time = unique_timestamps[train_cutoff_idx]
        val_cutoff_time   = unique_timestamps[val_cutoff_idx]

        train_df = df[df["timestamp"] < train_cutoff_time].copy()
        val_df   = df[(df["timestamp"] >= train_cutoff_time) & (df["timestamp"] < val_cutoff_time)].copy()
        test_df  = df[df["timestamp"] >= val_cutoff_time].copy()

        return train_df, val_df, test_df

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data quality and leakage checks.

        Returns:
            Dict containing validation status and metric summaries.
        """
        errors = []

        # Check 1: Duplicate (station_id, timestamp) rows
        dup_count = df.duplicated(subset=["station_id", "timestamp"]).sum()
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate (station_id, timestamp) rows.")

        # Check 2: Missing values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            errors.append(f"Found {nan_count} missing (NaN) values in dataset.")

        # Check 3: Occupancy bounds [0, 1]
        out_of_bounds_occ = ((df[self.TARGET_COL] < 0.0) | (df[self.TARGET_COL] > 1.0)).sum()
        if out_of_bounds_occ > 0:
            errors.append(f"Found {out_of_bounds_occ} occupancy_rate values outside [0, 1].")

        # Check 4: Slot capacity constraint
        invalid_slots = ((df["occupied_slots"] < 0) | (df["occupied_slots"] > df["total_slots"])).sum()
        if invalid_slots > 0:
            errors.append(f"Found {invalid_slots} rows where occupied_slots > total_slots or < 0.")

        # Check 5: Chronological ordering per station
        for station_id, group in df.groupby("station_id"):
            if not group["timestamp"].is_monotonic_increasing:
                errors.append(f"Station {station_id} timestamps are not strictly monotonically increasing.")

        # Check 6: Absence of target leakage from future rows
        # For any station, lag_1h at index i must match occupancy_rate at index i-1
        sample_station = df["station_id"].iloc[0]
        sample_group = df[df["station_id"] == sample_station].reset_index(drop=True)
        if len(sample_group) > 1:
            actual_lag = sample_group["lag_1h"].iloc[1]
            prev_occ   = sample_group[self.TARGET_COL].iloc[0]
            if abs(actual_lag - prev_occ) > 1e-5:
                errors.append(f"Target leakage detected: lag_1h ({actual_lag}) does not match previous target ({prev_occ}).")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "errors": errors,
            "total_rows": len(df),
            "unique_stations": df["station_id"].nunique(),
            "min_timestamp": df["timestamp"].min(),
            "max_timestamp": df["timestamp"].max(),
        }
