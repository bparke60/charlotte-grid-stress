"""Batch gradient descent for linear regression, independent portfolio implementation."""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def fit_gradient_descent(X, y, learning_rate=0.01, epochs=5000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    weights = np.zeros(design.shape[1])
    history = []
    for _ in range(epochs):
        residual = design @ weights - y
        gradient = (2 / len(y)) * design.T @ residual
        weights -= learning_rate * gradient
        history.append(float(np.mean((design @ weights - y) ** 2)))
    return weights, history

def predict(X, weights):
    return np.column_stack([np.ones(len(X)), np.asarray(X)]) @ weights

if __name__ == "__main__":
    df = load_diabetes(as_frame=True).frame
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns="target"), df["target"], test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    weights, history = fit_gradient_descent(scaler.transform(X_train), y_train)
    print("Training MSE:", round(history[0], 3), "->", round(history[-1], 3))
    print("Test RMSE:", round(float(np.sqrt(mean_squared_error(y_test, predict(scaler.transform(X_test), weights)))), 3))
