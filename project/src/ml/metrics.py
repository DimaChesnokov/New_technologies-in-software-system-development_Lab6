from __future__ import annotations
import numpy as np
from typing import Dict

# Считает MAE
def mae(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.mean(np.abs(y_true - y_pred)))

# Считает MSE
def mse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.mean((y_true - y_pred) ** 2))

# Считает RMSE
def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(mse(y_true, y_pred)))

# Считает MAPE в %
def mape(y_true: np.ndarray, y_pred: np.ndarray):
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

# Считает sMAPE в %
def smape(y_true: np.ndarray, y_pred: np.ndarray):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100)

# Возвращает словарь всех метрик
def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "SMAPE": smape(y_true, y_pred), "MAPE": mape(y_true, y_pred)}
