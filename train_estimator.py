import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import joblib

def main():
    print("Loading data...")
    df = pd.read_csv("../operator_sessions.csv")
    
    # Target and Features
    target = 'TaskDuration'
    
    le_task = LabelEncoder()
    df['TaskType_encoded'] = le_task.fit_transform(df['TaskType'])
    
    le_weather = LabelEncoder()
    df['Weather_encoded'] = le_weather.fit_transform(df['Weather'])
    
    le_soil = LabelEncoder()
    df['SoilType_encoded'] = le_soil.fit_transform(df['SoilType'])
    
    le_operator = LabelEncoder()
    df['Operator_encoded'] = le_operator.fit_transform(df['OperatorID'])
    
    FEATURES = [
        "TaskType_encoded", "Weather_encoded", "SoilType_encoded",
        "EngineHours", "Temperature", "WindSpeed", "ShiftNumber", "Operator_encoded", "OperatorFatigueScore"
    ]
    
    joblib.dump(le_task, 'le_task.pkl')
    joblib.dump(le_operator, 'le_operator.pkl')
    # weather and soil encoders are already saved from anomaly
    
    X = df[FEATURES]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "verbose": -1
    }
    
    print("Training Estimator model...")
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )
    
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"\nTest RMSE: {rmse:.2f} minutes")
    
    model.save_model("task_estimator.txt")
    print("Model saved to models/task_estimator.txt")

if __name__ == "__main__":
    main()
