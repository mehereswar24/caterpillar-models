# CAT Smart Operator — ML Models

Three trained models for the CAT Smart Operator platform.

## Models

| Model | Algorithm | File | Metric |
|---|---|---|---|
| Task Time Predictor | Random Forest + XGBoost Ensemble | `task_time_model.joblib` + `xgb_task_model.joblib` | MAE 20 min (XGB), 25 min (RF) |
| Anomaly Classifier | LightGBM | `anomaly_model.joblib` | F1-macro 0.9947 |
| Maintenance Predictor | LightGBM | `maintenance_model.joblib` | MAE 81 hours |

## Anomaly Classes (7)
`NORMAL`, `EXCESSIVE_IDLE`, `OVER_REV`, `HIGH_PRESSURE`, `OVERHEAT`, `SEATBELT_VIOLATION`, `PROXIMITY_BREACH`

## Dataset
- `data/tasks.csv` — 2,000 rows, 27 columns
- `data/machine_logs.csv` — 2,000 rows, 21 columns

## Retrain

```bash
pip install scikit-learn xgboost lightgbm pandas numpy joblib
python generate_data.py
python train_task.py
python train_anomaly.py
python train_maintenance.py
```

## API Endpoints (served via caterpillar-stack)

```
POST https://caterpillar-stack.onrender.com/api/predict
POST https://caterpillar-stack.onrender.com/api/anomaly/detect
GET  https://caterpillar-stack.onrender.com/api/anomaly/scan
POST https://caterpillar-stack.onrender.com/api/maintenance/predict
GET  https://caterpillar-stack.onrender.com/api/maintenance/status
```
