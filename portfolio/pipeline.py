"""Reproducible educational ETL + regression benchmark."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def load_and_validate():
    df = load_diabetes(as_frame=True).frame.copy()
    assert df.shape == (442, 11)
    assert df.columns.is_unique and not df.isna().any().any()
    return df

def store_and_reload(df, db_path="diabetes.db"):
    with sqlite3.connect(db_path) as conn:
        df.to_sql("diabetes_observations", conn, if_exists="replace", index=False)
        restored = pd.read_sql_query("SELECT * FROM diabetes_observations", conn)
    assert restored.shape == df.shape
    return restored

def evaluate(df):
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    train, test = scaler.transform(X_train), scaler.transform(X_test)
    results = []
    for name, model in [("Linear regression", LinearRegression()), ("Ridge regression", Ridge(alpha=10))]:
        model.fit(train, y_train)
        prediction = model.predict(test)
        results.append({"model": name,
                        "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 3),
                        "r2": round(float(r2_score(y_test, prediction)), 3)})
    return pd.DataFrame(results)

if __name__ == "__main__":
    data = store_and_reload(load_and_validate())
    print(evaluate(data).to_string(index=False))
