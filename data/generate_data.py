# ChargeFlow AI -- Synthetic Dataset Generator
# ============================================
# Generates three realistic datasets for the Indian EV charging ecosystem:
#   1. stations.csv        -- 50 charging stations across 5 Indian cities
#   2. sessions.csv        -- 10,000 historical charging sessions
#   3. realtime_status.csv -- Live snapshot of station status (for simulation)
#
# Author : ChargeFlow AI Team
# Event  : ETAuto Tech Hackathon 2026
# Theme  : Seamless EV Charging Ecosystem

import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants — Indian EV Ecosystem Context ───────────────────────────────────

CITIES = {
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "ev_density": 0.85, "n_stations": 14},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "ev_density": 0.90, "n_stations": 12},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "ev_density": 0.80, "n_stations": 10},
    "Pune":      {"lat": 18.5204, "lon": 73.8567, "ev_density": 0.70, "n_stations": 8},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "ev_density": 0.65, "n_stations": 6},
}

# Real Indian CPO (Charging Point Operators)
OPERATORS = [
    "Tata Power EZ Charge",
    "ChargeZone",
    "Ather Grid",
    "BPCL EV Station",
    "NTPC Vidyut",
    "Statiq",
    "Bolt.Earth",
    "Zeon Charging",
]

CHARGER_TYPES = {
    "AC Type 2 (7.4 kW)":  {"power_kw": 7.4,  "session_hrs": (0.75, 3.0),  "pct": 0.35},
    "AC Type 2 (22 kW)":   {"power_kw": 22.0, "session_hrs": (0.5, 1.5),   "pct": 0.25},
    "DC CCS2 (50 kW)":     {"power_kw": 50.0, "session_hrs": (0.33, 0.75), "pct": 0.20},
    "DC CCS2 (150 kW)":    {"power_kw": 150.0,"session_hrs": (0.2, 0.5),   "pct": 0.12},
    "DC CHAdeMO (50 kW)":  {"power_kw": 50.0, "session_hrs": (0.33, 0.75), "pct": 0.08},
}

VEHICLE_TYPES = [
    "Tata Nexon EV",
    "MG ZS EV",
    "Hyundai Kona Electric",
    "Ather 450X",
    "Ola S1 Pro",
    "Tata Tigor EV",
    "Mahindra XUV400",
    "BYD Atto 3",
    "Kia EV6",
    "BMW iX",
]

AMENITIES = [
    "Restroom", "Cafe", "WiFi", "Parking Shade", "24x7 Support",
    "Shopping Mall", "Security Camera", "EV Service Centre",
]

# Demand profile multipliers (24 hours) — India-specific peak patterns
# Peaks: morning commute (8-9), lunch (12-13), evening rush (18-21)
HOURLY_DEMAND = np.array([
    0.15, 0.10, 0.08, 0.07, 0.08, 0.12,   # 00–05
    0.20, 0.45, 0.85, 0.70, 0.55, 0.60,   # 06–11
    0.80, 0.65, 0.55, 0.50, 0.55, 0.65,   # 12–17
    0.90, 1.00, 0.95, 0.80, 0.55, 0.30,   # 18–23
])

# Weekend demand is ~25% lower overall but more uniform
WEEKEND_MULTIPLIER = np.array([
    0.20, 0.15, 0.10, 0.08, 0.10, 0.15,
    0.25, 0.35, 0.50, 0.65, 0.75, 0.80,
    0.85, 0.85, 0.80, 0.75, 0.70, 0.75,
    0.80, 0.85, 0.75, 0.60, 0.40, 0.25,
])

# ── Helper Functions ──────────────────────────────────────────────────────────

def jitter(lat: float, lon: float, radius_km: float = 15.0):
    """Add geographic jitter within a city radius."""
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * np.cos(np.radians(lat)))
    return (
        lat + np.random.uniform(-delta_lat, delta_lat),
        lon + np.random.uniform(-delta_lon, delta_lon),
    )

def pick_charger_type():
    """Sample charger type by realistic market share distribution."""
    types = list(CHARGER_TYPES.keys())
    weights = [CHARGER_TYPES[t]["pct"] for t in types]
    return random.choices(types, weights=weights, k=1)[0]

def tariff_per_kwh(charger_type: str, city: str) -> float:
    """Realistic tariff in INR/kWh — fast chargers cost more."""
    base = {
        "AC Type 2 (7.4 kW)":  12.0,
        "AC Type 2 (22 kW)":   14.0,
        "DC CCS2 (50 kW)":     18.0,
        "DC CCS2 (150 kW)":    22.0,
        "DC CHAdeMO (50 kW)":  18.0,
    }
    # Metro premium: Bengaluru and Delhi slightly higher
    metro_premium = 1.10 if city in ("Bengaluru", "Delhi") else 1.0
    return round(base[charger_type] * metro_premium + np.random.uniform(-1, 1), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — STATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_stations() -> pd.DataFrame:
    """
    50 EV charging stations across 5 Indian cities.
    Schema:
        station_id, name, city, latitude, longitude, total_slots,
        charger_type, operator, avg_power_kw, installation_year,
        amenities_score, amenities, tariff_per_kwh, is_24x7,
        has_fast_charger, parking_fee_per_hr
    """
    print("Generating stations.csv ...")
    records = []
    station_id = 1

    for city, props in CITIES.items():
        for i in range(props["n_stations"]):
            lat, lon = jitter(props["lat"], props["lon"])
            c_type = pick_charger_type()
            c_info = CHARGER_TYPES[c_type]
            n_slots = random.choice([2, 4, 4, 6, 6, 8, 10, 12])
            op = random.choice(OPERATORS)
            amenities_selected = random.sample(AMENITIES, k=random.randint(2, 5))
            amenities_score = round(len(amenities_selected) / len(AMENITIES) * 10, 1)
            install_year = random.choice([2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024])

            records.append({
                "station_id":        f"STA{station_id:03d}",
                "name":              f"{op} — {city} Station {i+1}",
                "city":              city,
                "latitude":          round(lat, 6),
                "longitude":         round(lon, 6),
                "total_slots":       n_slots,
                "charger_type":      c_type,
                "operator":          op,
                "avg_power_kw":      c_info["power_kw"],
                "installation_year": install_year,
                "amenities":         ", ".join(amenities_selected),
                "amenities_score":   amenities_score,
                "tariff_per_kwh":    tariff_per_kwh(c_type, city),
                "is_24x7":           random.choice([True, True, False]),
                "has_fast_charger":  c_info["power_kw"] >= 50,
                "parking_fee_per_hr": random.choice([0, 0, 20, 30, 50]),
            })
            station_id += 1

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "stations.csv", index=False)
    print(f"  [OK] stations.csv -- {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — SESSIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sessions(stations_df: pd.DataFrame, n_sessions: int = 10000) -> pd.DataFrame:
    """
    10,000 historical charging sessions over 90 days (Jan–Mar 2025).
    Demand follows realistic Indian EV usage patterns with:
      - Morning peak (8–9 AM), Lunch dip, Evening peak (6–9 PM)
      - Weekend vs weekday split
      - City-level demand density differences

    Schema:
        session_id, station_id, city, charger_type, start_time, end_time,
        duration_hrs, energy_kwh, vehicle_type, day_of_week, hour,
        is_weekend, wait_time_mins, revenue_inr, operator, user_segment
    """
    print("Generating sessions.csv ...")
    records = []

    start_date = pd.Timestamp("2025-01-01")
    end_date   = pd.Timestamp("2025-03-31")
    date_range = pd.date_range(start_date, end_date, freq="D")

    station_list = stations_df.to_dict("records")

    for _ in range(n_sessions):
        # Sample a date, then an hour weighted by demand profile
        date = random.choice(date_range)
        is_weekend = date.dayofweek >= 5
        demand_profile = WEEKEND_MULTIPLIER if is_weekend else HOURLY_DEMAND

        hour = random.choices(range(24), weights=demand_profile, k=1)[0]
        minute = random.randint(0, 59)
        start_time = date + pd.Timedelta(hours=hour, minutes=minute)

        # Sample station — weight by city EV density
        city_weights = [CITIES[s["city"]]["ev_density"] for s in station_list]
        station = random.choices(station_list, weights=city_weights, k=1)[0]

        # Session duration depends on charger type
        c_info = CHARGER_TYPES.get(station["charger_type"], CHARGER_TYPES["AC Type 2 (7.4 kW)"])
        min_h, max_h = c_info["session_hrs"]
        duration_hrs = round(np.random.uniform(min_h, max_h), 2)
        end_time = start_time + pd.Timedelta(hours=duration_hrs)

        energy_kwh = round(station["avg_power_kw"] * duration_hrs * np.random.uniform(0.75, 1.0), 2)
        revenue_inr = round(energy_kwh * station["tariff_per_kwh"], 2)

        # Wait time correlates with hour demand
        base_wait = HOURLY_DEMAND[hour] * 20  # max ~20 mins at peak
        wait_time_mins = max(0, round(np.random.normal(base_wait, 5), 1))

        user_segments = ["Daily Commuter", "Intercity Traveller", "Fleet Driver", "Weekend User"]
        user_seg_weights = [0.45, 0.15, 0.25, 0.15]

        records.append({
            "session_id":    f"SES{len(records)+1:06d}",
            "station_id":    station["station_id"],
            "city":          station["city"],
            "charger_type":  station["charger_type"],
            "operator":      station["operator"],
            "start_time":    start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time":      end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hrs":  duration_hrs,
            "energy_kwh":    energy_kwh,
            "vehicle_type":  random.choice(VEHICLE_TYPES),
            "day_of_week":   date.day_name(),
            "hour":          hour,
            "is_weekend":    is_weekend,
            "wait_time_mins": wait_time_mins,
            "revenue_inr":   revenue_inr,
            "user_segment":  random.choices(user_segments, weights=user_seg_weights, k=1)[0],
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "sessions.csv", index=False)
    print(f"  [OK] sessions.csv -- {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET 3 — REAL-TIME STATUS (Snapshot for Simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_realtime_status(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Live snapshot of all 50 stations at a simulated 'current' moment.
    Designed to be regenerated every N seconds in the live demo to
    simulate real-time IoT telemetry from OCPP-compliant chargers.

    Schema:
        station_id, city, operator, charger_type, total_slots,
        available_slots, occupied_slots, queue_length, current_load_kw,
        estimated_wait_mins, utilization_pct, status, last_updated
    """
    print("Generating realtime_status.csv ...")
    records = []
    now = pd.Timestamp.now()

    # Simulate 6 PM demand — near-peak conditions
    peak_hour = 18
    demand_factor = HOURLY_DEMAND[peak_hour]

    for _, sta in stations_df.iterrows():
        total = sta["total_slots"]
        city_factor = CITIES[sta["city"]]["ev_density"]

        # Occupied slots follow demand + some noise
        occupied = int(np.clip(
            np.random.binomial(total, demand_factor * city_factor),
            0, total
        ))
        available = total - occupied
        queue = max(0, int(np.random.poisson(occupied * 0.3)))

        c_info = CHARGER_TYPES.get(sta["charger_type"], CHARGER_TYPES["AC Type 2 (7.4 kW)"])
        min_h, max_h = c_info["session_hrs"]
        avg_session_min = ((min_h + max_h) / 2) * 60

        # M/M/c queue approximation for wait time
        if available > 0:
            estimated_wait = 0
        else:
            estimated_wait = round((queue / max(total, 1)) * avg_session_min, 1)

        load_kw = round(occupied * sta["avg_power_kw"] * np.random.uniform(0.8, 1.0), 1)
        utilization_pct = round((occupied / total) * 100, 1)

        if utilization_pct >= 90:
            status = "CRITICAL"
        elif utilization_pct >= 70:
            status = "BUSY"
        elif utilization_pct >= 40:
            status = "MODERATE"
        else:
            status = "AVAILABLE"

        records.append({
            "station_id":          sta["station_id"],
            "name":                sta["name"],
            "city":                sta["city"],
            "operator":            sta["operator"],
            "charger_type":        sta["charger_type"],
            "total_slots":         total,
            "available_slots":     available,
            "occupied_slots":      occupied,
            "queue_length":        queue,
            "current_load_kw":     load_kw,
            "estimated_wait_mins": estimated_wait,
            "utilization_pct":     utilization_pct,
            "status":              status,
            "latitude":            sta["latitude"],
            "longitude":           sta["longitude"],
            "last_updated":        now.strftime("%Y-%m-%d %H:%M:%S"),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "realtime_status.csv", index=False)
    print(f"  [OK] realtime_status.csv -- {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(stations_df, sessions_df, realtime_df):
    print("\n" + "="*60)
    print("  ChargeFlow AI -- Dataset Generation Summary")
    print("="*60)
    print(f"  Stations   : {len(stations_df)} across {stations_df['city'].nunique()} cities")
    print(f"  Sessions   : {len(sessions_df)} over 90 days (Jan-Mar 2025)")
    print(f"  Operators  : {stations_df['operator'].nunique()} unique CPOs")
    print(f"  Avg Wait   : {sessions_df['wait_time_mins'].mean():.1f} mins (all-day avg)")
    print(f"  Peak Wait  : {sessions_df[sessions_df['hour'].between(18,21)]['wait_time_mins'].mean():.1f} mins (evening peak)")
    print(f"  Avg Util   : {realtime_df['utilization_pct'].mean():.1f}% (current snapshot)")
    print(f"  Total Rev  : INR {sessions_df['revenue_inr'].sum():,.0f} (90-day period)")
    print(f"  Avg kWh    : {sessions_df['energy_kwh'].mean():.1f} kWh per session")
    print("="*60)
    print("  All files saved to:", OUTPUT_DIR.resolve())
    print("="*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nChargeFlow AI -- Generating synthetic datasets...\n")
    stations_df  = generate_stations()
    sessions_df  = generate_sessions(stations_df)
    realtime_df  = generate_realtime_status(stations_df)
    print_summary(stations_df, sessions_df, realtime_df)
