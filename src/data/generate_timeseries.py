"""
ChargeFlow AI V2 — Hourly Time-Series Data Generator
======================================================
Generates realistic station-level hourly EV charging telemetry:
- 50 stations across 5 Indian cities (using metadata from data/stations.csv)
- Hourly observations across configurable time horizon (default: 180 days)
- Realistic temporal demand patterns (morning/evening peaks, weekend shifts,
  city variations, charger-type profiles, seasonal temperature dynamics, noise)
- Physical occupancy bounds: 0 <= occupied_slots <= total_slots
- Physical occupancy rate: occupancy_rate = occupied_slots / total_slots
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

# Set up paths relative to project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# Major Indian Public Holidays in 2025 (H1 & H2)
INDIAN_HOLIDAYS_2025 = {
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Maha Shivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-ul-Fitr
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # May Day
    "2025-06-07",  # Bakrid / Id-ul-Zuha
    "2025-07-06",  # Muharram
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-20",  # Diwali
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
}

# City EV density multipliers
CITY_MULTIPLIERS = {
    "Bengaluru": 1.15,
    "Delhi":     1.10,
    "Mumbai":    1.05,
    "Pune":      0.95,
    "Hyderabad": 0.90,
}

# Charger type demand multipliers (fast chargers draw higher peak utilization)
CHARGER_MULTIPLIERS = {
    "DC CCS2 (150 kW)":   1.18,
    "DC CCS2 (50 kW)":    1.08,
    "AC Type 2 (22 kW)":  0.95,
    "AC Type 2 (7.4 kW)": 0.85,
    "DC CHAdeMO (50 kW)": 0.80,
}

# Diurnal hourly base occupancy probability curve (24 hours)
# Peak 1: Morning Commute (07:00–09:00, peak at 08:00)
# Peak 2: Evening Rush (18:00–21:00, peak at 19:00)
HOURLY_BASE_DEMAND = np.array([
    0.10, 0.08, 0.06, 0.05, 0.06, 0.10,   # 00:00 - 05:00 (Night off-peak)
    0.25, 0.60, 0.85, 0.70, 0.50, 0.55,   # 06:00 - 11:00 (Morning Peak)
    0.65, 0.55, 0.50, 0.48, 0.55, 0.70,   # 12:00 - 17:00 (Mid-day steady)
    0.90, 0.95, 0.85, 0.70, 0.45, 0.20    # 18:00 - 23:00 (Evening Peak)
])

# Weekend diurnal demand curve (delayed morning start, sustained afternoon/evening leisure)
HOURLY_WEEKEND_DEMAND = np.array([
    0.12, 0.08, 0.06, 0.05, 0.05, 0.08,   # 00:00 - 05:00
    0.15, 0.25, 0.45, 0.60, 0.70, 0.75,   # 06:00 - 11:00
    0.80, 0.80, 0.75, 0.70, 0.75, 0.85,   # 12:00 - 17:00 (Mid-day leisure peak)
    0.90, 0.85, 0.70, 0.50, 0.35, 0.20    # 18:00 - 23:00
])

# Monthly baseline temperatures in Celsius (Indian Climate Profile)
MONTHLY_BASE_TEMP = {
    1: 20.5, 2: 24.0, 3: 28.5, 4: 33.5, 5: 36.0, 6: 31.5,
    7: 28.0, 8: 27.5, 9: 28.0, 10: 27.0, 11: 23.5, 12: 20.0
}


def generate_hourly_timeseries(
    stations_csv_path: str = None,
    num_days: int = 180,
    start_date: str = "2025-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate station-level hourly EV charging time-series data.

    Args:
        stations_csv_path: Path to stations.csv (defaults to data/stations.csv).
        num_days: Total number of days to generate (default 180).
        start_date: Start date string (YYYY-MM-DD).
        seed: Random seed for exact reproducibility.

    Returns:
        DataFrame with required hourly time-series columns.
    """
    np.random.seed(seed)
    random.seed(seed)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if stations_csv_path is None:
        stations_csv_path = DATA_DIR / "stations.csv"
    else:
        stations_csv_path = Path(stations_csv_path)

    if not stations_csv_path.exists():
        raise FileNotFoundError(f"Stations metadata not found at {stations_csv_path}")

    stations_df = pd.read_csv(stations_csv_path)
    
    # Generate timestamp range (1-hour frequency)
    timestamps = pd.date_range(
        start=pd.Timestamp(start_date),
        periods=num_days * 24,
        freq="h"
    )

    records = []

    for _, sta in stations_df.iterrows():
        station_id = sta["station_id"]
        city = sta["city"]
        charger_type = sta["charger_type"]
        total_slots = int(sta["total_slots"])
        avg_power_kw = float(sta["avg_power_kw"])

        city_mult = CITY_MULTIPLIERS.get(city, 1.0)
        charger_mult = CHARGER_MULTIPLIERS.get(charger_type, 1.0)

        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            date_str = ts.strftime("%Y-%m-%d")
            is_weekend = ts.dayofweek >= 5
            is_holiday = date_str in INDIAN_HOLIDAYS_2025

            # Select diurnal demand profile
            if is_weekend or is_holiday:
                base_p = HOURLY_WEEKEND_DEMAND[hour]
            else:
                base_p = HOURLY_BASE_DEMAND[hour]

            # Calculate ambient temperature with diurnal oscillation (hottest around 14:00)
            diurnal_temp_var = 4.5 * np.sin(np.pi * (hour - 9) / 12.0)
            ambient_temp_c = round(
                MONTHLY_BASE_TEMP.get(month, 25.0) + diurnal_temp_var + np.random.normal(0, 1.2),
                1
            )

            # Temperature effect factor (slightly higher battery cooling / AC load in extreme heat)
            temp_effect = 1.0 + 0.003 * max(0, ambient_temp_c - 28.0)

            # Target expected occupancy probability
            p_target = np.clip(
                base_p * city_mult * charger_mult * temp_effect + np.random.normal(0, 0.035),
                0.02,
                0.98
            )

            # Sample physical occupied slots using Binomial distribution
            occupied_slots = int(np.random.binomial(n=total_slots, p=p_target))
            occupied_slots = int(np.clip(occupied_slots, 0, total_slots))

            # Physical occupancy rate
            occupancy_rate = round(occupied_slots / float(total_slots), 4)

            # Calculate actual grid load (kW)
            if occupied_slots > 0:
                load_efficiency = np.random.uniform(0.85, 0.98)
                grid_load_kw = round(occupied_slots * avg_power_kw * load_efficiency, 2)
            else:
                grid_load_kw = 0.0

            records.append({
                "station_id":      station_id,
                "timestamp":       ts,
                "city":            city,
                "charger_type":    charger_type,
                "total_slots":     total_slots,
                "occupied_slots":  occupied_slots,
                "occupancy_rate":  occupancy_rate,
                "grid_load_kw":    grid_load_kw,
                "temperature_c":   ambient_temp_c,
                "is_weekend":      is_weekend,
                "is_holiday":      is_holiday,
            })

    df = pd.DataFrame(records)
    # Ensure correct sorting: station_id then timestamp
    df = df.sort_values(by=["station_id", "timestamp"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = ARTIFACTS_DIR / "hourly_charging_data.csv"
    print(f"Generating hourly time-series data (180 days, 50 stations)...")
    df_ts = generate_hourly_timeseries(num_days=180, seed=42)
    df_ts.to_csv(out_file, index=False)
    print(f"Dataset successfully created and saved to {out_file}")
    print(f"Shape: {df_ts.shape[0]:,} rows × {df_ts.shape[1]} columns")
    print(f"Date range: {df_ts['timestamp'].min()} to {df_ts['timestamp'].max()}")
