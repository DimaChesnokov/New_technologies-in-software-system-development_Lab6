from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Возвращает результат ADF-теста на стационарность
def adf_test(y: pd.Series) -> dict:
    stat, pval, usedlag, nobs, crit, icbest = adfuller(y.dropna().values, autolag="AIC")
    return {"ADF": float(stat), "pvalue": float(pval), "lags": int(usedlag),
            "nobs": int(nobs), "crit": {k: float(v) for k, v in crit.items()}}

# Рисует ACF/PACF и показывает окно
def show_acf_pacf(y: pd.Series, lags: int = 40) -> None:
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121); plot_acf(y.dropna(), ax=ax1, lags=lags)
    ax2 = fig.add_subplot(122); plot_pacf(y.dropna(), ax=ax2, lags=lags)
    fig.tight_layout(); plt.show()
