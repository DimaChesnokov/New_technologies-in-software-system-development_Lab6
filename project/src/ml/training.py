from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from .metrics import all_metrics
from .persistence import save_model

@dataclass(frozen=True)
class SplitCfg:
    # Хвост теста в точках
    test_size: int

# Делит ряд на train/test по времени
def time_split(y: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    return y.iloc[:-test_size], y.iloc[-test_size:]

# Логирует один прогон в CSV
def log_run(log_csv: Path, row: Dict[str, Any]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if log_csv.exists():
        df.to_csv(log_csv, mode="a", index=False, header=False)
    else:
        df.to_csv(log_csv, index=False)

# Возвращает таблицу прогонов, отсортированную по метрике
def select_best(log_csv: Path, metric: str = "RMSE") -> pd.DataFrame:
    df = pd.read_csv(log_csv)
    return df.sort_values(metric).reset_index(drop=True)

# Обучает и валидирует SARIMA
def train_eval_sarima(y: pd.Series, order, seasonal_order, split: SplitCfg,
                      run_name: str, out_dir: Path) -> Dict[str, Any]:
    from .ts_models import SarimaWrapper
    train, test = time_split(y, split.test_size)
    m = SarimaWrapper(order, seasonal_order).fit(train)
    pred = m.predict(start=len(train), end=len(train)+len(test)-1)
    met = all_metrics(test.values, pred.values)
    meta = {"algo": "SARIMA", "order": order, "seasonal_order": seasonal_order,
            "train_len": len(train), "test_len": len(test), **met}
    log_run(out_dir / "logs" / "lr6_runs.csv", {"run": run_name, **meta})
    return {"model": m, "metrics": met, "meta": meta}

# Обучает и валидирует регрессор по окнам
def train_eval_regressor(X: pd.DataFrame, y: pd.Series, model_kind: str, model_kwargs: Dict[str, Any],
                         split: SplitCfg, run_name: str, out_dir: Path) -> Dict[str, Any]:
    from .ml_models import SkModel
    n = len(y)
    train_idx = np.arange(0, n - split.test_size)
    test_idx = np.arange(n - split.test_size, n)
    m = SkModel(model_kind, **model_kwargs).fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
    pred = m.predict(X.iloc[test_idx].values)
    met = all_metrics(y.iloc[test_idx].values, pred)
    meta = {"algo": model_kind, **m.get_params(), "train_len": len(train_idx),
            "test_len": len(test_idx), **met}
    log_run(out_dir / "logs" / "lr6_runs.csv", {"run": run_name, **meta})
    return {"model": m, "metrics": met, "meta": meta}

# Сохраняет лучшую модель и метаданные
def persist_best(model_obj, meta: Dict[str, Any], out_dir: Path) -> None:
    save_model(model_obj, meta, out_dir / "models" / "best_model.pkl",
               out_dir / "models" / "best_meta.json")
