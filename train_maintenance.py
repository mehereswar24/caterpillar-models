"""
train_maintenance.py — Train LightGBM maintenance predictor.
Predicts hours_until_service from machine telemetry.
Output: maintenance_model.joblib + maintenance_feature_cols.json
"""
import json, os, joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

DIR  = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data", "machine_logs.csv")

df = pd.read_csv(DATA)

# Target: hours until next 500h service
SERVICE_INTERVAL = 500
df["hours_until_service"] = SERVICE_INTERVAL - (df["engine_hours"] % SERVICE_INTERVAL)

# Feature engineering
df["idle_ratio"]      = df["idle_time_min"] / (df["idle_time_min"] + df["active_time_min"] + 1)
df["rpm_x_pressure"]  = df["rpm"] * df["hydraulic_pressure"] / 1000
df["fuel_per_active"] = df["fuel_used_l"] / (df["active_time_min"] + 1)
df["weather_enc"]     = pd.Categorical(df["weather"]).codes
df["ground_enc"]      = pd.Categorical(df["ground_condition"]).codes
df["fault_enc"]       = (df["fault_codes"] != "NONE").astype(int)

FEATURES = ["engine_hours","rpm","hydraulic_pressure","temperature_c",
            "fuel_level","fuel_used_l","idle_time_min","active_time_min",
            "engine_load_pct","tilt_angle","speed_kph",
            "idle_ratio","rpm_x_pressure","fuel_per_active",
            "weather_enc","ground_enc","fault_enc"]

X = df[FEATURES].fillna(0)
y = df["hours_until_service"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                      random_state=42, n_jobs=-1, verbosity=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds)**0.5

print(f"LightGBM Maintenance — MAE: {mae:.2f} hours, RMSE: {rmse:.2f} hours")
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(FEATURES)}")

joblib.dump(model, os.path.join(DIR, "maintenance_model.joblib"))
with open(os.path.join(DIR, "maintenance_feature_cols.json"), "w") as f:
    json.dump(FEATURES, f)

print("Saved: maintenance_model.joblib")
