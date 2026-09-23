# CAT Smart Operator — ML Models

Training scripts, datasets, and trained models for the CAT Smart Operator platform.

## Models

| Model | Algorithm | File | Performance |
|---|---|---|---|
| Task Time Predictor | Random Forest | `task_time_model.joblib` | MAE 25 min |
| Task Time Predictor | XGBoost | `xgb_task_model.joblib` | MAE 20 min |
| Anomaly Classifier | LightGBM — 7 classes | `anomaly_model.joblib` | F1-macro 0.9947 |
| Maintenance Predictor | LightGBM regressor | `maintenance_model.joblib` | MAE 81 hours |
| Fuel Sufficiency | LightGBM classifier | `fuel_model.joblib` | F1 0.9925 |
| Fuel Remaining | LightGBM regressor | `fuel_range_model.joblib` | MAE 2.87 L |

## Anomaly Classes (7)
`NORMAL` `EXCESSIVE_IDLE` `OVER_REV` `HIGH_PRESSURE` `OVERHEAT` `SEATBELT_VIOLATION` `PROXIMITY_BREACH`

## Datasets

| File | Rows | Cols | Used for |
|---|---|---|---|
| `data/tasks.csv` | 2,000 | 27 | Task time prediction |
| `data/machine_logs.csv` | 2,000 | 21 | Anomaly + maintenance |
| `data/fuel_data.csv` | 3,000 | 17 | Fuel sufficiency |

## Setup & Retrain

```bash
pip install scikit-learn xgboost lightgbm pandas numpy joblib

python generate_data.py        # generates tasks.csv + machine_logs.csv
python generate_fuel_data.py   # generates fuel_data.csv

python train_task.py           # RF + XGBoost
python train_anomaly.py        # LightGBM anomaly
python train_maintenance.py    # LightGBM maintenance
python train_fuel.py           # LightGBM fuel
```

## API Endpoints

Models are served via **[caterpillar-stack](https://github.com/mehereswar24/caterpillar-stack)**:

```
POST https://caterpillar-stack.onrender.com/api/predict
POST https://caterpillar-stack.onrender.com/api/anomaly/detect
GET  https://caterpillar-stack.onrender.com/api/anomaly/scan
POST https://caterpillar-stack.onrender.com/api/maintenance/predict
GET  https://caterpillar-stack.onrender.com/api/maintenance/status
POST https://caterpillar-stack.onrender.com/api/fuel/check
GET  https://caterpillar-stack.onrender.com/api/fuel/status
```
