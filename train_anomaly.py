"""
train_anomaly.py — Train LightGBM anomaly classifier on machine telemetry.
7 classes: NORMAL, EXCESSIVE_IDLE, OVER_REV, HIGH_PRESSURE, OVERHEAT,
           SEATBELT_VIOLATION, PROXIMITY_BREACH
Output: anomaly_model.joblib + anomaly_label_encoder.joblib + anomaly_feature_cols.json
"""
import json, os, joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report

DIR  = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data", "machine_logs.csv")

df = pd.read_csv(DATA, parse_dates=["timestamp"])

# Feature engineering
df["idle_ratio"]      = df["idle_time_min"] / (df["idle_time_min"] + df["active_time_min"] + 1)
df["hour"]            = df["timestamp"].dt.hour
df["is_night"]        = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)
df["seatbelt_enc"]    = (df["seatbelt"] == "unfastened").astype(int)
df["rpm_x_pressure"]  = df["rpm"] * df["hydraulic_pressure"] / 1000
df["fuel_per_active"] = df["fuel_used_l"] / (df["active_time_min"] + 1)
df["weather_enc"]     = pd.Categorical(df["weather"]).codes
df["ground_enc"]      = pd.Categorical(df["ground_condition"]).codes

def label_row(r):
    if r["seatbelt_enc"] == 1 and r["speed_kph"] > 1: return "SEATBELT_VIOLATION"
    if r["proximity_alert"] == 1:                      return "PROXIMITY_BREACH"
    if r["rpm"] > 2100:                                return "OVER_REV"
    if r["hydraulic_pressure"] > 270:                  return "HIGH_PRESSURE"
    if r["temperature_c"] > 98:                        return "OVERHEAT"
    if r["idle_ratio"] > 0.5 and r["active_time_min"] > 0: return "EXCESSIVE_IDLE"
    return "NORMAL"

df["label"] = df.apply(label_row, axis=1)
print("Label distribution:\n", df["label"].value_counts().to_string())

FEATURES = ["rpm","hydraulic_pressure","temperature_c","fuel_level","fuel_used_l",
            "idle_time_min","active_time_min","speed_kph","tilt_angle","engine_load_pct",
            "idle_ratio","hour","is_night","seatbelt_enc","proximity_alert",
            "rpm_x_pressure","fuel_per_active","weather_enc","ground_enc"]

X = df[FEATURES].fillna(0)
le = LabelEncoder()
y  = le.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LGBMClassifier(n_estimators=400, max_depth=8, learning_rate=0.05,
                       class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
f1 = f1_score(y_test, preds, average="macro")

print(f"\nF1-macro: {f1:.4f}")
print(classification_report(y_test, preds, target_names=le.classes_))

joblib.dump(model, os.path.join(DIR, "anomaly_model.joblib"))
joblib.dump(le,    os.path.join(DIR, "anomaly_label_encoder.joblib"))
with open(os.path.join(DIR, "anomaly_feature_cols.json"), "w") as f:
    json.dump(FEATURES, f)

print(f"Saved: anomaly_model.joblib | Classes: {list(le.classes_)}")
