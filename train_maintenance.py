import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def assign_component(row):
    """Rule-based component label derived from telemetry — gives the model real signal."""
    hours_since_service = row['EngineHours'] - row['LastServiceHours']
    if row['HydraulicPressure'] > 260 or (row['FaultCodes'] != 'NONE' and 'HYD' in str(row['FaultCodes'])):
        return 'hydraulic'
    if row['RPM'] > 2000 or hours_since_service > 450:
        return 'engine'
    if row['LoadCycles'] > 80 or row['ArmCycles'] > 200:
        return 'tracks'
    if hours_since_service > 350:
        return 'filter'
    return 'coolant'

def main():
    print("Loading data...")
    df = pd.read_csv("../operator_sessions.csv")

    # Target 1: Hours until next service (regression)
    df['HoursUntilService'] = np.clip(500 - (df['EngineHours'] - df['LastServiceHours']), 0, 500)

    # Target 2: Component at risk — deterministic from telemetry, not random
    df['ComponentAtRisk'] = df.apply(assign_component, axis=1)
    print("Component distribution:\n", df['ComponentAtRisk'].value_counts())

    le_comp = LabelEncoder()
    df['Component_encoded'] = le_comp.fit_transform(df['ComponentAtRisk'])
    joblib.dump(le_comp, 'le_component.pkl')

    FEATURES = [
        "EngineHours", "LastServiceHours", "HydraulicPressure",
        "RPM", "FuelUsed", "LoadCycles", "ArmCycles"
    ]

    X = df[FEATURES]
    y_reg = df['HoursUntilService']
    y_clf = df['Component_encoded']

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    # --- Regressor ---
    print("Training hours-until-service regressor...")
    td_reg = lgb.Dataset(X_train, label=yr_train)
    vd_reg = lgb.Dataset(X_test,  label=yr_test, reference=td_reg)
    params_reg = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 63, "learning_rate": 0.05, "verbose": -1,
    }
    m_reg = lgb.train(params_reg, td_reg, num_boost_round=300,
                      valid_sets=[vd_reg],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    mae = mean_absolute_error(yr_test, m_reg.predict(X_test))
    print(f"MAE (hours): {mae:.1f} hours")
    m_reg.save_model("maintenance_hours_predictor.txt")

    # --- Classifier ---
    print("Training component-at-risk classifier...")
    td_clf = lgb.Dataset(X_train, label=yc_train)
    vd_clf = lgb.Dataset(X_test,  label=yc_test, reference=td_clf)
    params_clf = {
        "objective": "multiclass", "num_class": len(le_comp.classes_),
        "metric": "multi_logloss", "num_leaves": 63,
        "learning_rate": 0.05, "verbose": -1,
    }
    m_clf = lgb.train(params_clf, td_clf, num_boost_round=300,
                      valid_sets=[vd_clf],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
    yc_pred = np.argmax(m_clf.predict(X_test), axis=1)
    acc = accuracy_score(yc_test, yc_pred)
    print(f"Accuracy (component): {acc:.2f}")
    print(classification_report(yc_test, yc_pred, target_names=le_comp.classes_))
    m_clf.save_model("maintenance_component_predictor.txt")

    print("Done — both maintenance models saved.")

if __name__ == "__main__":
    main()
