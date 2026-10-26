import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline import load_and_validate, store_and_reload, evaluate
from gradient_descent import fit_gradient_descent
from python_algorithms import center_list, matrix_replacement, histogram, flatten

def test_data_shape():
    assert load_and_validate().shape == (442, 11)

def test_sql_roundtrip(tmp_path):
    df = load_and_validate()
    assert store_and_reload(df, str(tmp_path / "test.db")).shape == df.shape

def test_regression_metrics():
    assert evaluate(load_and_validate())["rmse"].gt(0).all()

def test_gradient_descent_reduces_loss():
    X = np.arange(10).reshape(-1, 1) / 10
    y = 1 + 2 * X.ravel()
    _, history = fit_gradient_descent(X, y, epochs=1000)
    assert history[-1] < history[0]

def test_center_list():
    mean, centered = center_list([1, 2, 3])
    assert mean == 2 and centered == [-1, 0, 1]

def test_matrix_replacement():
    assert matrix_replacement([[1, None], [None, 4]]) == [[1, 0], [0, 4]]

def test_histogram():
    assert histogram(["a", "a", "b"]) == {"a": 2, "b": 1}

def test_flatten():
    assert flatten([1, [2, [3]]]) == [1, 2, 3]
