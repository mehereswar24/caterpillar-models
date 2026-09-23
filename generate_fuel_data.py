"""
generate_fuel_data.py — Generate fuel sufficiency dataset.
Features: fuel_level_pct, fuel_used_per_hour_l, task_duration_est_min,
          distance_to_station_km, tank_capacity_l, engine_load_pct,
          temperature_c, ground_condition, task_type
Target: can_complete (1/0) + fuel_after_task_l + range_km
"""
import pandas as pd
import numpy as np
import os

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

TANK_CAPACITY = 410  # CAT 320 tank capacity in litres
BASE_CONSUMPTION = 18.5  # litres/hour base

task_types   = ["Earth Excavation","Trenching","Material Loading","Grading","Compaction","Demolition"]
grounds      = ["Dry","Wet","Muddy","Frozen"]

rows = []
for i in range(3000):
    fuel_pct        = round(float(rng.uniform(5, 100)), 1)
    engine_load     = round(float(rng.uniform(40, 98)), 1)
    temperature     = round(float(rng.uniform(5, 45)), 1)
    ground          = rng.choice(grounds)
    task_type       = rng.choice(task_types)
    task_dur_min    = round(float(rng.uniform(20, 300)), 1)
    distance_km     = round(float(rng.uniform(0.5, 50)), 2)

    # Fuel consumption model
    load_factor   = 0.8 + (engine_load / 100) * 0.6
    temp_factor   = 1.0 + max(0, temperature - 35) * 0.008
    ground_factor = {"Dry": 1.0, "Wet": 1.08, "Muddy": 1.18, "Frozen": 1.12}[ground]
    task_factor   = {"Earth Excavation":1.1,"Trenching":1.15,"Material Loading":0.95,
                     "Grading":1.0,"Compaction":0.90,"Demolition":1.25}[task_type]

    consumption_per_hour = BASE_CONSUMPTION * load_factor * temp_factor * ground_factor * task_factor
    consumption_per_hour = round(float(consumption_per_hour + rng.normal(0, 0.5)), 2)

    fuel_available_l  = round((fuel_pct / 100) * TANK_CAPACITY, 1)
    task_dur_hr       = task_dur_min / 60
    fuel_needed_l     = round(consumption_per_hour * task_dur_hr, 1)
    fuel_after_l      = round(max(0, fuel_available_l - fuel_needed_l), 1)
    fuel_after_pct    = round((fuel_after_l / TANK_CAPACITY) * 100, 1)

    # Range calculation (km) at current consumption
    range_km = round((fuel_available_l / consumption_per_hour) * 8.5, 1)  # 8.5 km/h avg travel speed

    # Can complete: needs enough fuel for task + 10% reserve
    reserve_l   = TANK_CAPACITY * 0.10
    can_complete = 1 if fuel_available_l >= (fuel_needed_l + reserve_l) else 0

    # Can reach station: range > distance
    can_reach_station = 1 if range_km >= distance_km else 0

    # Risk level
    if not can_complete and not can_reach_station:
        risk = "CRITICAL"
    elif not can_complete:
        risk = "HIGH"
    elif fuel_pct < 25:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    rows.append({
        "log_id": i+1,
        "fuel_level_pct": fuel_pct,
        "fuel_available_l": fuel_available_l,
        "engine_load_pct": engine_load,
        "temperature_c": temperature,
        "ground_condition": ground,
        "task_type": task_type,
        "task_duration_est_min": task_dur_min,
        "consumption_l_per_hour": consumption_per_hour,
        "fuel_needed_l": fuel_needed_l,
        "fuel_after_task_l": fuel_after_l,
        "fuel_after_task_pct": fuel_after_pct,
        "range_km": range_km,
        "distance_to_station_km": distance_km,
        "can_complete_task": can_complete,
        "can_reach_station": can_reach_station,
        "risk_level": risk,
    })

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/fuel_data.csv", index=False)
print(f"fuel_data.csv → {len(df)} rows, {len(df.columns)} columns")
print("Risk distribution:\n", df["risk_level"].value_counts().to_string())
print("can_complete rate:", df["can_complete_task"].mean().round(3))
