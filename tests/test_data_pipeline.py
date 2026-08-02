"""
ChargeFlow AI V2 — Data Pipeline Unit Tests
=============================================
Built-in unittest test suite verifying:
1. Occupancy rate bounds [0.0, 1.0]
2. Slot capacity constraints (0 <= occupied_slots <= total_slots)
3. Strict chronological train/validation/test splitting
4. Expected feature column generation
5. Absence of target leakage from future rows
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.generate_timeseries import generate_hourly_timeseries
from src.data.preprocessor import DataPreprocessor


class TestDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate a small 14-day dataset for fast unit testing."""
        cls.raw_timeseries_df = generate_hourly_timeseries(num_days=14, seed=42)
        cls.preprocessor = DataPreprocessor()
        cls.featured_df = cls.preprocessor.engineer_features(cls.raw_timeseries_df)
        cls.train_df, cls.val_df, cls.test_df = cls.preprocessor.split_chronological(cls.featured_df)

    def test_occupancy_bounds(self):
        """Verify that occupancy_rate is bounded in [0.0, 1.0]."""
        occ_rate = self.raw_timeseries_df["occupancy_rate"]
        self.assertTrue((occ_rate >= 0.0).all(), "Occupancy rate contains negative values!")
        self.assertTrue((occ_rate <= 1.0).all(), "Occupancy rate exceeds 1.0!")
        
        # Verify exact physical definition: occupancy_rate == occupied_slots / total_slots
        expected_rate = (self.raw_timeseries_df["occupied_slots"] / self.raw_timeseries_df["total_slots"]).round(4)
        np.testing.assert_allclose(occ_rate.values, expected_rate.values, atol=1e-4)

    def test_slot_capacity_constraints(self):
        """Verify that occupied_slots is an integer between 0 and total_slots."""
        occupied = self.raw_timeseries_df["occupied_slots"]
        total = self.raw_timeseries_df["total_slots"]

        self.assertTrue((occupied >= 0).all(), "Occupied slots contains negative values!")
        self.assertTrue((occupied <= total).all(), "Occupied slots exceeds total slots capacity!")
        self.assertTrue(np.issubdtype(occupied.dtype, np.integer), "Occupied slots must be integer!")

    def test_expected_feature_columns(self):
        """Verify that feature engineering creates all required columns."""
        expected_cols = DataPreprocessor.FEATURE_COLS + [DataPreprocessor.TARGET_COL]

        for col in expected_cols:
            self.assertIn(col, self.featured_df.columns, f"Missing expected feature column: {col}")

        self.assertEqual(self.featured_df.isna().sum().sum(), 0, "Preprocessed dataset contains unexpected NaN values!")

    def test_chronological_splitting(self):
        """Verify strict chronological ordering across train, validation, and test splits."""
        max_train_time = self.train_df["timestamp"].max()
        min_val_time   = self.val_df["timestamp"].min()
        max_val_time   = self.val_df["timestamp"].max()
        min_test_time  = self.test_df["timestamp"].min()

        self.assertLess(max_train_time, min_val_time, f"Leakage: max train time ({max_train_time}) >= min val time ({min_val_time})")
        self.assertLess(max_val_time, min_test_time, f"Leakage: max val time ({max_val_time}) >= min test time ({min_test_time})")

        total_rows = len(self.train_df) + len(self.val_df) + len(self.test_df)
        self.assertGreater(total_rows, 0, "Splits are empty!")

    def test_absence_of_target_leakage(self):
        """Verify that lag and rolling features use ONLY past timestamps."""
        # Pick a specific station and inspect sequential rows
        station_id = self.featured_df["station_id"].iloc[0]
        sta_df = self.featured_df[self.featured_df["station_id"] == station_id].reset_index(drop=True)

        for idx in range(1, min(10, len(sta_df))):
            # lag_1h at index idx MUST equal occupancy_rate at index idx-1
            expected_lag_1h = sta_df[DataPreprocessor.TARGET_COL].iloc[idx - 1]
            actual_lag_1h   = sta_df["lag_1h"].iloc[idx]
            self.assertAlmostEqual(actual_lag_1h, expected_lag_1h, places=5,
                msg=f"Target leakage at index {idx}: lag_1h ({actual_lag_1h}) != prev target ({expected_lag_1h})")

            # rolling_mean_6h at index idx MUST equal mean of occupancy_rate from idx-6 to idx-1
            if idx >= 6:
                past_6_occ = sta_df[DataPreprocessor.TARGET_COL].iloc[idx-6:idx]
                expected_roll_mean = past_6_occ.mean()
                actual_roll_mean = sta_df["rolling_mean_6h"].iloc[idx]
                self.assertAlmostEqual(actual_roll_mean, expected_roll_mean, places=5,
                    msg=f"Target leakage at index {idx}: rolling_mean_6h ({actual_roll_mean}) != past mean ({expected_roll_mean})")

    def test_data_validation_suite(self):
        """Verify that DataPreprocessor.validate_dataset reports valid dataset status."""
        validation_res = self.preprocessor.validate_dataset(self.featured_df)

        self.assertTrue(validation_res["is_valid"], f"Validation failed with errors: {validation_res['errors']}")
        self.assertEqual(len(validation_res["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
