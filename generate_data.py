"""
generate_data.py — Generate training datasets for all 3 CAT Smart Operator models.
Output: data/tasks.csv (2000 rows, 27 cols) + data/machine_logs.csv (2000 rows, 21 cols)
"""
import pandas as pd
import numpy as np
import os

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

machine_ids  = [f"EXC{i:03d}" for i in range(1, 6)]
operator_ids = [f"OP{i:03d}"  for i in range(1, 11)]

# ── tasks.csv — for Task Time Predictor ──────────────────────────────────────
task_types   = ["Earth Excavation","Trenching","Material Loading","Grading","Compaction","Demolition"]
BASE = {"Earth Excavation":65,"Trenching":80,"Material Loading":35,"Grading":50,"Compaction":45,"Demolition":110}

rows = []
for i in range(1, 2001):
    task_type  = rng.choice(task_types)
    complexity = rng.choice(["Low","Medium","High"])
    material   = rng.choice(["Soil","Clay","Rock","Gravel","Sand","Mixed"])
    weather    = rng.choice(["Sunny","Cloudy","Rainy","Windy"])
    ground     = rng.choice(["Dry","Wet","Muddy","Frozen"])
    skill      = rng.choice(["Beginner","Intermediate","Expert"])
    qty        = round(float(rng.uniform(30,250)),1)
    depth      = round(float(rng.uniform(0.5,6.0)),1)
    haul       = round(float(rng.uniform(5,100)),1)
    temp       = round(float(rng.uniform(5,42)),1)
    rainfall   = round(float(rng.uniform(0,50) if weather=="Rainy" else 0),1)
    wind       = round(float(rng.uniform(10,60) if weather=="Windy" else rng.uniform(0,15)),1)
    op_exp     = round(float(rng.uniform(0.5,15)),1)
    mach_age   = round(float(rng.uniform(0.5,15)),1)
    eng_hrs    = round(float(rng.uniform(200,12000)),0)
    bucket     = round(float(rng.choice([0.8,1.0,1.2,1.5,2.0])),1)
    eng_load   = round(float(rng.uniform(50,95)),1)
    mach_eff   = round(float(rng.uniform(65,98)),1)
    prev_tasks = int(rng.integers(0,300))
    prev_avg   = round(float(BASE[task_type]+rng.normal(0,8)),1)

    base = BASE[task_type]
    actual = (base * (qty/100)**0.6 * (1+(depth-2)*0.05) * (1+(haul-25)*0.003) * (1.2/bucket)
              * {"Low":0.85,"Medium":1.0,"High":1.25}[complexity]
              * {"Soil":1.0,"Sand":0.95,"Gravel":1.05,"Clay":1.15,"Mixed":1.1,"Rock":1.35}[material]
              * {"Sunny":1.0,"Cloudy":1.03,"Rainy":1.20,"Windy":1.08}[weather]
              * {"Dry":1.0,"Wet":1.12,"Muddy":1.25,"Frozen":1.18}[ground]
              * (1+rainfall*0.003) * (1+max(0,wind-20)*0.004)
              * {"Beginner":1.25,"Intermediate":1.0,"Expert":0.82}[skill]
              * max(0.75,1.15-op_exp*0.025) * (1+mach_age*0.015)
              * (1+(90-mach_eff)*0.005) * (1+max(0,eng_load-80)*0.003)
              * (max(0.85,1.0-prev_tasks*0.0005) if prev_tasks>10 else 1.0)
              + rng.normal(0,3))
    actual = max(10, min(300, round(float(actual),1)))
    estimated = max(5, round(float(actual+rng.normal(0,8)),1))

    rows.append({"task_id":i,"machine_id":rng.choice(machine_ids),"operator_id":rng.choice(operator_ids),
        "task_type":task_type,"task_complexity":complexity,"material_type":material,
        "quantity_m3":qty,"target_depth_m":depth,"haul_distance_m":haul,
        "weather":weather,"temperature_c":temp,"rainfall_mm":rainfall,"wind_speed_kmh":wind,
        "ground_condition":ground,"operator_skill":skill,"operator_experience_yrs":op_exp,
        "previous_similar_tasks":prev_tasks,"previous_avg_completion_min":prev_avg,
        "machine_age_yrs":mach_age,"engine_hours":eng_hrs,"bucket_capacity_m3":bucket,
        "avg_engine_load_pct":eng_load,"machine_efficiency_pct":mach_eff,
        "estimated_time_min":estimated,"actual_completion_time_min":actual,
        "completed":int(rng.random()>0.05),
        "date":str(pd.Timestamp("2024-01-01")+pd.Timedelta(days=int(rng.integers(0,365))))})

pd.DataFrame(rows).to_csv(f"{OUT}/tasks.csv", index=False)
print(f"tasks.csv → 2000 rows, 27 columns")

# ── machine_logs.csv — for Anomaly + Maintenance models ──────────────────────
logs = pd.DataFrame({
    "log_id":             range(1,2001),
    "machine_id":         rng.choice(machine_ids,2000),
    "operator_id":        rng.choice(operator_ids,2000),
    "timestamp":          pd.date_range("2024-01-01",periods=2000,freq="30min"),
    "engine_hours":       rng.uniform(100,12000,2000).round(1),
    "rpm":                rng.integers(600,2400,2000),
    "hydraulic_pressure": rng.uniform(100,320,2000).round(1),
    "fuel_level":         rng.uniform(5,100,2000).round(1),
    "fuel_used_l":        rng.uniform(0.5,18,2000).round(2),
    "idle_time_min":      rng.integers(0,90,2000),
    "active_time_min":    rng.integers(10,120,2000),
    "speed_kph":          rng.uniform(0,18,2000).round(1),
    "tilt_angle":         rng.uniform(0,22,2000).round(1),
    "temperature_c":      rng.uniform(30,110,2000).round(1),
    "engine_load_pct":    rng.uniform(40,98,2000).round(1),
    "seatbelt":           rng.choice(["fastened","unfastened"],2000,p=[0.88,0.12]),
    "proximity_alert":    rng.choice([0,1],2000,p=[0.85,0.15]),
    "fault_codes":        rng.choice(["NONE","P0100","P0217","P0562"],2000,p=[0.80,0.07,0.07,0.06]),
    "weather":            rng.choice(["Sunny","Cloudy","Rainy","Windy"],2000),
    "ground_condition":   rng.choice(["Dry","Wet","Muddy","Frozen"],2000),
    "shift":              rng.choice(["day","night"],2000),
})
for i in range(2000):
    if rng.random()<0.08: logs.at[i,"idle_time_min"]=int(rng.integers(60,90)); logs.at[i,"rpm"]=int(rng.integers(600,900))
    if rng.random()<0.05: logs.at[i,"rpm"]=int(rng.integers(2200,2400))
    if rng.random()<0.06: logs.at[i,"hydraulic_pressure"]=float(rng.uniform(290,320))
    if rng.random()<0.04: logs.at[i,"temperature_c"]=float(rng.uniform(100,110))

logs.to_csv(f"{OUT}/machine_logs.csv", index=False)
print(f"machine_logs.csv → 2000 rows, 21 columns")
