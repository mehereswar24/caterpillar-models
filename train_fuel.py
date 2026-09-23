"""
train_fuel.py — Train fuel sufficiency classifier + range regressor.
Models:
  1. LightGBM classifier → can_complete_task (0/1)
  2. LightGBM regressor  → fuel_after_task_l (how much fuel remains)
Output: fuel_model.joblib, fuel_range_model.joblib, fuel_feature_cols.json
"""
import json, os, joblib
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error, classification_report

DIR  = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data", "fuel_data.csv")

df = pd.read_csv(DATA)

# Encode categoricals
df["ground_enc"] = pd.Categorical(df["ground_condition"]).codes
df["task_enc"]   = pd.Categorical(df["task_type"]).codes

FEATURES = [
    "fuel_level_pct", "fuel_available_l", "engine_load_pct",
    "temperature_c", "task_duration_est_min", "consumption_l_per_hour",
    "distance_to_station_km", "ground_enc", "task_enc"
]

X = df[FEATURES].fillna(0)

# ── Model 1: can_complete_task classifier ────────────────────────────────────
y_cls = df["can_complete_task"]
X_train, X_test, y_train, y_test = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

clf = LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                     class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
f1 = f1_score(y_test, preds)
print(f"Fuel Classifier — F1: {f1:.4f}")
print(classification_report(y_test, preds, target_names=["INSUFFICIENT","SUFFICIENT"]))

joblib.dump(clf, os.path.join(DIR, "fuel_model.joblib"))

# ── Model 2: fuel_after_task regressor ───────────────────────────────────────
y_reg = df["fuel_after_task_l"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg = LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                    random_state=42, n_jobs=-1, verbosity=-1)
reg.fit(X_train2, y_train2)
mae = mean_absolute_error(y_test2, reg.predict(X_test2))
print(f"Fuel Regressor  — MAE: {mae:.2f} litres")

joblib.dump(reg, os.path.join(DIR, "fuel_range_model.joblib"))

with open(os.path.join(DIR, "fuel_feature_cols.json"), "w") as f:
    json.dump(FEATURES, f)

print("Saved: fuel_model.joblib + fuel_range_model.joblib + fuel_feature_cols.json")
