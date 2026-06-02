"""
ChargeFlow AI — Demand Predictor
=================================
Predicts 24-hour charger occupancy for a given station using
a Random Forest Regressor trained on historical session data.

Why Random Forest?
  - Handles mixed feature types (categorical + numerical) natively
  - Inherently interpretable via feature importance scores
  - No feature scaling required
  - Robust to noisy synthetic data
  - Fast training and inference (< 1 second on 10K rows)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


class DemandPredictor:
    """
    Predicts hourly occupancy rate (0.0–1.0) for an EV charging station.

    Usage:
        predictor = DemandPredictor()
        predictor.train(sessions_df, stations_df)
        forecast = predictor.predict_24hr(station_id="STA001", date="2025-04-01")
    """

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
        self.label_encoders = {}
        self.feature_cols = [
            "hour", "day_of_week_num", "is_weekend",
            "city_enc", "charger_type_enc", "total_slots",
            "avg_power_kw", "amenities_score",
        ]
        self.is_trained = False

    def _encode(self, df: pd.DataFrame, col: str, fit: bool = False) -> pd.Series:
        """Label-encode a categorical column."""
        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        else:
            le = self.label_encoders[col]
            encoded = le.transform(df[col].astype(str))
        return encoded

    def _build_features(self, sessions_df: pd.DataFrame,
                        stations_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Merge sessions with station metadata and engineer features."""
        df = sessions_df.merge(stations_df, on="station_id", how="left")
        df["start_time"] = pd.to_datetime(df["start_time"])
        df["day_of_week_num"] = df["start_time"].dt.dayofweek

        # Occupancy rate: normalize wait time as proxy for demand
        max_wait = df["wait_time_mins"].max()
        df["occupancy_rate"] = (df["wait_time_mins"] / max_wait).clip(0, 1)

        df["city_enc"] = self._encode(df, "city", fit=fit)
        df["charger_type_enc"] = self._encode(df, "charger_type_x", fit=fit)

        return df

    def train(self, sessions_df: pd.DataFrame, stations_df: pd.DataFrame):
        """Train the demand prediction model."""
        df = self._build_features(sessions_df, stations_df, fit=True)

        X = df[self.feature_cols].fillna(0)
        y = df["occupancy_rate"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  Demand Predictor — MAE: {mae:.4f}, R²: {r2:.4f}")

        self.is_trained = True
        self.stations_df = stations_df
        return {"mae": mae, "r2": r2}

    def predict_24hr(self, station_id: str, date: str) -> pd.DataFrame:
        """
        Forecast occupancy rate for each hour of a given date.

        Returns:
            DataFrame with columns: hour, predicted_occupancy, lower_bound, upper_bound
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        sta = self.stations_df[self.stations_df["station_id"] == station_id].iloc[0]
        target_date = pd.to_datetime(date)
        is_weekend = target_date.dayofweek >= 5
        day_of_week_num = target_date.dayofweek

        city_enc = self.label_encoders["city"].transform([sta["city"]])[0]
        charger_enc = self.label_encoders["charger_type_x"].transform([sta["charger_type"]])[0]

        rows = []
        for hour in range(24):
            X = pd.DataFrame([{
                "hour":              hour,
                "day_of_week_num":   day_of_week_num,
                "is_weekend":        int(is_weekend),
                "city_enc":          city_enc,
                "charger_type_enc":  charger_enc,
                "total_slots":       sta["total_slots"],
                "avg_power_kw":      sta["avg_power_kw"],
                "amenities_score":   sta["amenities_score"],
            }])

            # Use individual tree predictions for confidence interval
            tree_preds = np.array([
                tree.predict(X)[0] for tree in self.model.estimators_
            ])
            pred = tree_preds.mean()
            lower = np.percentile(tree_preds, 10)
            upper = np.percentile(tree_preds, 90)

            rows.append({
                "hour":                hour,
                "predicted_occupancy": round(float(pred), 3),
                "lower_bound":         round(float(lower), 3),
                "upper_bound":         round(float(upper), 3),
            })

        return pd.DataFrame(rows)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores for explainability."""
        return pd.DataFrame({
            "feature":   self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
