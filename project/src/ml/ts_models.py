from __future__ import annotations
import pandas as pd
from typing import Tuple, Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaWrapper:
    # Инициализирует параметры SARIMA
    def __init__(self, order: Tuple[int,int,int],
                 seasonal_order: Tuple[int,int,int,int] = (0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.res_ = None

    # Обучает модель на ряду
    def fit(self, y: pd.Series) -> "SarimaWrapper":
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        self.res_ = self.model.fit(disp=False)
        return self

    # Делает прогноз по индексам начала/конца
    def predict(self, start: int, end: int) -> pd.Series:
        return self.res_.predict(start=start, end=end)

    # Возвращает параметры модели
    def get_params(self) -> Dict[str, Any]:
        return {"order": self.order, "seasonal_order": self.seasonal_order}
