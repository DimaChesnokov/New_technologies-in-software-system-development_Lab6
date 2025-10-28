from __future__ import annotations
import pandas as pd
from typing import Tuple

# Делает датасет «скользящее окно» для регрессии
def make_supervised(df: pd.DataFrame, date_col: str, value_col: str,
                    lag_k: int, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    s = df[value_col].astype(float).reset_index(drop=True)
    data = {}
    for i in range(1, lag_k + 1):
        data[f"lag_{i}"] = s.shift(i)
    d = pd.to_datetime(df[date_col]).reset_index(drop=True)
    data["dow"] = d.dt.dayofweek
    data["dom"] = d.dt.day
    data["month"] = d.dt.month
    X = pd.DataFrame(data).iloc[lag_k:-horizon].reset_index(drop=True)
    y = s.shift(-horizon).iloc[lag_k:-horizon].reset_index(drop=True)
    return X, y
