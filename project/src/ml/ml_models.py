from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class SkModel:
    # Создаёт sklearn-модель по ключу
    def __init__(self, kind: str, **kwargs: Any):
        if kind == "linear":
            self.model = LinearRegression(**kwargs)
        elif kind == "rf":
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError("Unsupported model kind")
        self.kind = kind

    # Обучает модель
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    # Предсказывает значения
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # Возвращает параметры модели
    def get_params(self) -> Dict[str, Any]:
        return {"kind": self.kind, **self.model.get_params()}
