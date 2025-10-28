from __future__ import annotations
from PySide6 import QtWidgets
from pathlib import Path
import pandas as pd
from src.ml.training import (SplitCfg, train_eval_sarima, train_eval_regressor,
                             select_best, persist_best)
from src.ml.features import make_supervised



class TabLR6(QtWidgets.QWidget):
    # Инициализирует вкладку ЛР-6
    def __init__(self, data_csv: Path, parent=None):
        super().__init__(parent)
        self.data_csv = data_csv
        self._build_ui()

    # Строит элементы интерфейса
    def _build_ui(self):
        self.btn_run = QtWidgets.QPushButton("Обучить и проверить")
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.out)
        self.btn_run.clicked.connect(self._run)

    # Пишет строку в лог окна
    def _append(self, txt: str):
        self.out.appendPlainText(txt)

    # Выполняет обучение двух моделей и сохранение лучшей
    def _run(self):
        df = pd.read_csv(self.data_csv)
        df.columns = ["date", "rate"]
        df["date"] = pd.to_datetime(df["date"])
        out_dir = Path(".")

        res1 = train_eval_sarima(df["rate"], order=(1,1,1), seasonal_order=(1,0,1,7),
                                 split=SplitCfg(test_size=30), run_name="sarima_1", out_dir=out_dir)
        self._append(f"SARIMA: {res1['metrics']}")

        X, y = make_supervised(df, "date", "rate", lag_k=7, horizon=1)
        res2 = train_eval_regressor(X, y, model_kind="rf",
                                    model_kwargs={"n_estimators": 200, "random_state": 42},
                                    split=SplitCfg(test_size=30), run_name="rf_200", out_dir=out_dir)
        self._append(f"RF: {res2['metrics']}")

        best_tbl = select_best(Path("logs/lr6_runs.csv"))
        self._append(f"Лучшие запуски:\n{best_tbl.head(5).to_string(index=False)}")

        best = res1 if res1["metrics"]["RMSE"] <= res2["metrics"]["RMSE"] else res2
        persist_best(best["model"], best["meta"], out_dir)
        self._append("Лучшая модель сохранена: models/best_model.pkl")
