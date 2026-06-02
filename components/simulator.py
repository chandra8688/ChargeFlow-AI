"""
ChargeFlow AI — Live Status Simulator
========================================
Simulates real-time OCPP telemetry updates for the demo.
Refreshes realtime_status.csv every N seconds with slightly varied values,
mimicking live IoT feed from charging stations.

Used by the Streamlit app with streamlit-autorefresh for the live demo.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Hourly demand profile (same as data generator)
HOURLY_DEMAND = np.array([
    0.15, 0.10, 0.08, 0.07, 0.08, 0.12,
    0.20, 0.45, 0.85, 0.70, 0.55, 0.60,
    0.80, 0.65, 0.55, 0.50, 0.55, 0.65,
    0.90, 1.00, 0.95, 0.80, 0.55, 0.30,
])


def update_realtime_status(noise_scale: float = 0.08) -> pd.DataFrame:
    """
    Load current realtime_status.csv, apply small random changes,
    and save back. Simulates OCPP telemetry ticks.

    Args:
        noise_scale : Standard deviation of random perturbation (default 8%)

    Returns:
        Updated DataFrame
    """
    try:
        df = pd.read_csv(DATA_DIR / "realtime_status.csv")
        stations_df = pd.read_csv(DATA_DIR / "stations.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Run data/generate_data.py first to create datasets.")

    current_hour = pd.Timestamp.now().hour
    demand_factor = HOURLY_DEMAND[current_hour]

    for idx, row in df.iterrows():
        total = row["total_slots"]

        # Small stochastic delta to occupied slots
        delta = int(np.random.normal(0, noise_scale * total))
        occupied = int(np.clip(row["occupied_slots"] + delta, 0, total))
        available = total - occupied

        # Queue: occasionally someone joins or leaves
        queue_delta = np.random.choice([-1, 0, 0, 0, 1], p=[0.1, 0.5, 0.2, 0.1, 0.1])
        queue = max(0, int(row["queue_length"]) + queue_delta)

        # Wait time — from simple queuing
        charger_row = stations_df[stations_df["station_id"] == row["station_id"]]
        avg_power = charger_row["avg_power_kw"].values[0] if len(charger_row) else 22
        avg_session_mins = max(30, 22 * 60 / max(avg_power, 1))
        estimated_wait = 0 if available > 0 else round((queue / max(total, 1)) * avg_session_mins, 1)

        load_kw = round(occupied * avg_power * np.random.uniform(0.8, 1.0), 1)
        utilization_pct = round((occupied / total) * 100, 1)

        if utilization_pct >= 90:
            status = "CRITICAL"
        elif utilization_pct >= 70:
            status = "BUSY"
        elif utilization_pct >= 40:
            status = "MODERATE"
        else:
            status = "AVAILABLE"

        df.at[idx, "occupied_slots"]      = occupied
        df.at[idx, "available_slots"]     = available
        df.at[idx, "queue_length"]        = queue
        df.at[idx, "current_load_kw"]     = load_kw
        df.at[idx, "utilization_pct"]     = utilization_pct
        df.at[idx, "estimated_wait_mins"] = estimated_wait
        df.at[idx, "status"]              = status
        df.at[idx, "last_updated"]        = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(DATA_DIR / "realtime_status.csv", index=False)
    return df


def get_network_kpis(realtime_df: pd.DataFrame) -> dict:
    """
    Compute network-level KPIs from the current realtime snapshot.

    Returns:
        dict with total_stations, active_sessions, avg_utilization,
        critical_stations, available_stations, total_load_kw
    """
    return {
        "total_stations":    len(realtime_df),
        "active_sessions":   int(realtime_df["occupied_slots"].sum()),
        "avg_utilization":   round(realtime_df["utilization_pct"].mean(), 1),
        "critical_stations": int((realtime_df["status"] == "CRITICAL").sum()),
        "available_stations": int((realtime_df["available_slots"] > 0).sum()),
        "total_load_kw":     round(realtime_df["current_load_kw"].sum(), 1),
        "avg_wait_mins":     round(realtime_df["estimated_wait_mins"].mean(), 1),
    }
