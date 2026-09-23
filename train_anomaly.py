import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv("../operator_sessions.csv")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour_of_day'] = df['Timestamp'].dt.hour
    df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 18)).astype(int)

    # Add engineered features that separate anomaly classes cleanly
    df['idle_ratio'] = df['IdlingTime'] / (df['IdlingTime'] + df['ActiveTime'] + 1)
    df['rpm_x_pressure'] = df['RPM'] * df['HydraulicPressure']
    df['fuel_per_active_min'] = df['FuelUsed'] / (df['ActiveTime'] + 1)
    df['speed_x_unfastened'] = df['SpeedKPH'] * (df['SeatbeltStatus'] == 'UNFASTENED').astype(int)

    le_weather = LabelEncoder(); df['weather_encoded'] = le_weather.fit_transform(df['Weather'])
    le_soil    = LabelEncoder(); df['soil_encoded']    = le_soil.fit_transform(df['SoilType'])
    le_seatbelt= LabelEncoder(); df['seatbelt_encoded']= le_seatbelt.fit_transform(df['SeatbeltStatus'])
    le_target  = LabelEncoder(); df['target']          = le_target.fit_transform(df['AlertType'])

    joblib.dump(le_weather,  'le_weather.pkl')
    joblib.dump(le_soil,     'le_soil.pkl')
    joblib.dump(le_target,   'le_target.pkl')
    joblib.dump(le_seatbelt, 'le_seatbelt.pkl')

    FEATURES = [
        "RPM", "HydraulicPressure", "TiltAngle", "FuelUsed",
        "LoadCycles", "IdlingTime", "ActiveTime", "SpeedKPH", "EngineHours",
        "hour_of_day", "is_night", "weather_encoded", "soil_encoded", "seatbelt_encoded",
        # engineered
        "idle_ratio", "rpm_x_pressure", "fuel_per_active_min", "speed_x_unfastened",
    ]

    X = df[FEATURES]; y = df['target']

    # Class weights — inverse frequency so minority classes are amplified
    counts = np.bincount(y)
    weights = 1.0 / counts[y]
    weights = weights / weights.mean()   # normalise around 1

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data   = lgb.Dataset(X_test,  label=y_test,  reference=train_data)

    num_classes = len(le_target.classes_)

    params = {
        "objective":        "multiclass",
        "num_class":        num_classes,
        "metric":           "multi_logloss",
        "num_leaves":       127,
        "learning_rate":    0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples":5,
        "verbose":          -1,
    }

    print("Training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=800,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nTest F1 Score (Macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    model.save_model("anomaly_detector.txt")
    print("Saved → models/anomaly_detector.txt")

if __name__ == "__main__":
    main()
