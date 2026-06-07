"""Evaluation metrics for power-system inertia estimation."""

from __future__ import annotations

import numpy as np


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return signed relative error in percent."""
    return (y_pred - y_true) / (np.abs(y_true) + 1e-8) * 100.0


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute percentage error in percent."""
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)) * 100.0)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.05) -> float:
    """Return ACC, the percentage of predictions within a relative tolerance."""
    rel_abs_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)
    return float(np.mean(rel_abs_error <= tol) * 100.0)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.05) -> dict[str, float]:
    """Compute all metrics used by the training and evaluation scripts."""
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "ACC": accuracy(y_true, y_pred, tol=tol),
        "R2": r_squared(y_true, y_pred),
    }
