"""
train_task.py — Train RF + XGBoost ensemble for task time prediction.
Output: task_time_model.joblib (Random Forest) + xgb_task_model.joblib (XGBoost)
"""
import json, os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DIR  = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DIR, "data", "tasks.csv")
CATS = ["task_type","task_complexity","material_type","weather","ground_condition","operator_skill"]

df = pd.read_csv(DATA)
df.dropna(subset=["actual_completion_time_min"], inplace=True)
df_enc = pd.get_dummies(df, columns=CATS, drop_first=False)
feature_cols = [c for c in df_enc.columns if c not in
    ["task_id","machine_id","operator_id","date","actual_completion_time_min","estimated_time_min","completed"]]

X = df_enc[feature_cols].fillna(0)
y = df_enc["actual_completion_time_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_mae  = mean_absolute_error(y_test, rf.predict(X_test))
rf_rmse = mean_squared_error(y_test, rf.predict(X_test))**0.5
joblib.dump(rf, os.path.join(DIR, "task_time_model.joblib"))
print(f"Random Forest  — MAE: {rf_mae:.2f} min, RMSE: {rf_rmse:.2f} min")

# XGBoost
xgb = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_mae  = mean_absolute_error(y_test, xgb.predict(X_test))
xgb_rmse = mean_squared_error(y_test, xgb.predict(X_test))**0.5
joblib.dump(xgb, os.path.join(DIR, "xgb_task_model.joblib"))
print(f"XGBoost        — MAE: {xgb_mae:.2f} min, RMSE: {xgb_rmse:.2f} min")

with open(os.path.join(DIR, "feature_cols.json"), "w") as f:
    json.dump(feature_cols, f)

print(f"Features: {len(feature_cols)} | Train: {len(X_train)} | Test: {len(X_test)}")
