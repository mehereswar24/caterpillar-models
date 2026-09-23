# CAT Smart Operator — Models

Trained LightGBM models and label encoders for the CAT Smart Operator Assistant.

## Models

| File | Type | Metric |
|---|---|---|
| `anomaly_detector.txt` | LightGBM multiclass | F1-macro 0.994 |
| `task_estimator.txt` | LightGBM regression | RMSE 5.15 min |
| `maintenance_hours_predictor.txt` | LightGBM regression | MAE 1.2 h |
| `maintenance_component_predictor.txt` | LightGBM multiclass | Acc 0.94 |

## Label Encoders

`le_target.pkl`, `le_task.pkl`, `le_weather.pkl`, `le_soil.pkl`, `le_seatbelt.pkl`, `le_operator.pkl`, `le_component.pkl`

## Retrain from scratch

```bash
# Generate dataset
cd data && python generate_dataset.py

# Train all models
cd ../models
python train_anomaly.py
python train_estimator.py
python train_maintenance.py
```

## Dataset

`operator_sessions.csv` — 10,000 synthetic rows, 36 columns, 7 anomaly classes balanced (~500 each).
