from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATE_FMT = "%Y-%m-%d"

@dataclass(frozen=True)
class EdaConfig:
    date_col: str = "date"     
    value_col: str = "rate"  

# 1) чтение CSV DataFrame
def read_df(csv_path: Path, cfg: EdaConfig = EdaConfig()):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns] # 2) нижний регистр _
    if cfg.date_col not in df.columns or cfg.value_col not in df.columns:
        raise ValueError(f"Ожидаются колонки '{cfg.date_col}', '{cfg.value_col}'")
    df = df[[cfg.date_col, cfg.value_col]].copy()
    # 3)  nan/non
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], format=DATE_FMT, errors="raise")
    df[cfg.value_col] = pd.to_numeric(df[cfg.value_col], errors="coerce") 
    df = df.dropna(subset=[cfg.value_col]).drop_duplicates(subset=[cfg.date_col]).sort_values(cfg.date_col)
    return df.reset_index(drop=True)

# 4) добавление отклонений от медианы и среднего
def add_deviation_cols(df: pd.DataFrame, cfg: EdaConfig = EdaConfig()):
    # столбцы: dev_from_median, dev_from_mean
    v = df[cfg.value_col]
    df = df.copy()
    df["dev_from_median"] = (v - v.median()).abs()
    df["dev_from_mean"] = (v - v.mean()).abs()

    df["dev_norm"] = df["dev_from_mean"] / (v.std(ddof=1) 
                                            if len(v) > 1 
                                            else 1.0)
    
    
    return df

# 5) статистика + boxplot
def describe_with_outliers(df: pd.DataFrame, cfg: EdaConfig = EdaConfig()):
    need_cols = [cfg.value_col] + [c for c in ("dev_from_median", "dev_from_mean") if c in df.columns]
    desc = df[need_cols].describe()
    q1, q3 = df[cfg.value_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    fences = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    plt.figure()
    plt.boxplot(df[cfg.value_col].values, vert=False, showmeans=True)
    plt.title("Boxplot курса")
    plt.ylabel(cfg.value_col)
    plt.show()
    return desc, fences

# 6) фильтр по отклонению от среднего
def filter_by_deviation(df: pd.DataFrame, threshold: float, cfg: EdaConfig = EdaConfig()):
    mean_v = df[cfg.value_col].mean()
    mask = (df[cfg.value_col] - mean_v).abs() >= threshold
    return df.loc[mask].reset_index(drop=True)

# 7) фильтр по диапазону дат
def filter_by_date_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: EdaConfig = EdaConfig()):
    d = df[cfg.date_col]
    return df.loc[(d >= start.normalize()) & (d <= end.normalize())].reset_index(drop=True)

# 8) группировка по месяцу со средним значением курса
def groupby_month_mean(df: pd.DataFrame, cfg: EdaConfig = EdaConfig()) -> pd.DataFrame:
    g = df.copy()
    g["month"] = g[cfg.date_col].dt.to_period("M")
    res = g.groupby("month", as_index=False)[cfg.value_col].mean()
    res["month"] = res["month"].astype(str)
    return res

# 9) график всего периода
def plot_full_period(df: pd.DataFrame, cfg: EdaConfig = EdaConfig()):
    plt.figure()
    plt.plot(df[cfg.date_col], df[cfg.value_col])
    plt.title("Курс за весь период")
    plt.xlabel("Дата")
    plt.ylabel(cfg.value_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 10) график за месяц с медианой и средним
def plot_month(df: pd.DataFrame, year: int, month: int, cfg: EdaConfig = EdaConfig()):
    mdf = df[(df[cfg.date_col].dt.year == year) & (df[cfg.date_col].dt.month == month)]
    if mdf.empty:
        raise ValueError("Нет данных за указанный месяц")
    median_v = mdf[cfg.value_col].median()
    mean_v = mdf[cfg.value_col].mean()
    plt.figure()
    plt.plot(mdf[cfg.date_col], mdf[cfg.value_col], marker="o")
    plt.axhline(median_v, linestyle="--", label=f"median={median_v:.4f}")
    plt.axhline(mean_v, linestyle=":", label=f"mean={mean_v:.4f}")
    plt.title(f"Курс за {year:04d}-{month:02d}")
    plt.xlabel("Дата")
    plt.ylabel(cfg.value_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
