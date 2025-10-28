from __future__ import annotations
from pathlib import Path
import pandas as pd
from PySide6 import QtWidgets, QtCore
import numpy as np
from src.dataset_io import save_XY
from src.splitters import split_by_year, split_by_week
from src.source import Source
from src.query import get_value
from src.annotate import annotate_csv, annotate_dir
from pathlib import Path
from app.tab_lr6 import TabLR6


from src.eda import *

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab3+Lab4 KZT→RUB GUI")
        self.resize(980, 560)

        
        self.input_dir: Path | None = None
        self.input_csv: Path | None = None
        self.output_dir: Path | None = None

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

       
        tab_lr3 = QtWidgets.QWidget()
        tabs.addTab(tab_lr3, "ЛР-3: подготовка")
        v = QtWidgets.QVBoxLayout(tab_lr3)

        grp_src = QtWidgets.QGroupBox("Исходный датасет")
        v.addWidget(grp_src)
        g = QtWidgets.QGridLayout(grp_src)

        self.le_input_dir = QtWidgets.QLineEdit(); self.le_input_dir.setReadOnly(True)
        btn_pick_dir = QtWidgets.QPushButton("Выбрать папку…")
        btn_pick_dir.clicked.connect(self.pick_input_dir)

        self.cb_csv = QtWidgets.QComboBox()
        self.cb_csv.currentIndexChanged.connect(self._on_csv_change)

        g.addWidget(QtWidgets.QLabel("Папка:"), 0, 0)
        g.addWidget(self.le_input_dir, 0, 1)
        g.addWidget(btn_pick_dir, 0, 2)
        g.addWidget(QtWidgets.QLabel("CSV:"), 1, 0)
        g.addWidget(self.cb_csv, 1, 1, 1, 2)

        grp_actions = QtWidgets.QGroupBox("Операции")
        v.addWidget(grp_actions)
        h = QtWidgets.QHBoxLayout(grp_actions)

        btn_annot_src = QtWidgets.QPushButton("Аннотация исходного CSV…")
        btn_annot_src.clicked.connect(self.make_src_annotation)
        btn_build = QtWidgets.QPushButton("Собрать датасет (X/Y, годы, недели) + аннотации…")
        btn_build.clicked.connect(self.build_datasets)

        h.addWidget(btn_annot_src)
        h.addWidget(btn_build)
        h.addStretch(1)

        grp_query = QtWidgets.QGroupBox("Запрос по дате")
        v.addWidget(grp_query)
        ql = QtWidgets.QGridLayout(grp_query)

        self.date_edit = QtWidgets.QDateEdit(QtCore.QDate.currentDate())
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setCalendarPopup(True)

        self.src_mode = QtWidgets.QComboBox()
        self.src_mode.addItems(["RAW", "BY_YEAR", "BY_WEEK"])

        self.le_query_root = QtWidgets.QLineEdit(); self.le_query_root.setReadOnly(True)
        btn_pick_query_root = QtWidgets.QPushButton("Папка источника…")
        btn_pick_query_root.clicked.connect(self.pick_query_root)

        btn_get = QtWidgets.QPushButton("Получить данные")
        btn_get.clicked.connect(self.do_query)
        self.lbl_result = QtWidgets.QLabel("—")

        ql.addWidget(QtWidgets.QLabel("Дата:"), 0, 0)
        ql.addWidget(self.date_edit, 0, 1)
        ql.addWidget(QtWidgets.QLabel("Источник:"), 0, 2)
        ql.addWidget(self.src_mode, 0, 3)
        ql.addWidget(self.le_query_root, 1, 0, 1, 3)
        ql.addWidget(btn_pick_query_root, 1, 3)
        ql.addWidget(btn_get, 2, 0)
        ql.addWidget(self.lbl_result, 2, 1, 1, 3)

        v.addStretch(1)

       
        tabs.addTab(EdaTab(self), "ЛР-4: EDA")
        # ЛР-6: новая вкладка с моделями
        tabs.addTab(TabLR6(Path("data/dataset_KZT.csv")), "ЛР-6: Модели и прогноз")

    # ЛР-3
    def pick_input_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с исходным датасетом")
        if not path:
            return
        self.input_dir = Path(path)
        self.le_input_dir.setText(path)
        self.cb_csv.clear()
        csvs = sorted(self.input_dir.glob("*.csv"))
        if not csvs:
            QtWidgets.QMessageBox.warning(self, "Нет CSV", "В выбранной папке нет *.csv")
            return
        for p in csvs:
            self.cb_csv.addItem(p.name, str(p))
        self._on_csv_change()

    def _on_csv_change(self):
        idx = self.cb_csv.currentIndex()
        self.input_csv = None if idx < 0 else Path(self.cb_csv.currentData())

    def make_src_annotation(self):
        if self.input_csv is None:
            QtWidgets.QMessageBox.warning(self, "Не выбран CSV", "Сначала выберите исходный CSV.")
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Куда сохранить аннотацию", "annotation_src.json", "JSON (*.json)")
        if not out:
            return
        annotate_csv(self.input_csv, Path(out))
        QtWidgets.QMessageBox.information(self, "Готово", f"Аннотация сохранена:\n{out}")

    def build_datasets(self):
        if self.input_csv is None:
            QtWidgets.QMessageBox.warning(self, "Не выбран CSV", "Сначала выберите исходный CSV.")
            return
        dest = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку назначения")
        if not dest:
            return
        self.output_dir = Path(dest)

        save_XY(self.input_csv, self.output_dir / "X.csv", self.output_dir / "Y.csv")
        by_year = self.output_dir / "by_year"; by_year.mkdir(parents=True, exist_ok=True)
        by_week = self.output_dir / "by_week"; by_week.mkdir(parents=True, exist_ok=True)
        split_by_year(self.input_csv, by_year)
        split_by_week(self.input_csv, by_week)

        annotate_csv(self.input_csv, self.output_dir / "annotation_src.json")
        annotate_dir(by_year, self.output_dir / "annotation_by_year.json")
        annotate_dir(by_week, self.output_dir / "annotation_by_week.json")

        QtWidgets.QMessageBox.information(self, "Готово", f"Данные собраны в:\n{dest}")

    def pick_query_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Папка источника для запроса")
        if not path:
            return
        self.le_query_root.setText(path)

    def do_query(self):
        mode = self.src_mode.currentText()
        root = self.le_query_root.text().strip()
        d = pd.Timestamp(self.date_edit.date().toString("yyyy-MM-dd"))
        src_path = Path(root)

        if mode == "RAW":
            csvs = sorted(src_path.glob("*.csv"))
            if not csvs:
                QtWidgets.QMessageBox.warning(self, "Нет CSV", "В папке нет *.csv для RAW.")
                return
            src = Source.raw(csvs[0])
        elif mode == "BY_YEAR":
            src = Source.by_year(src_path)
        else:
            src = Source.by_week(src_path)

        res = get_value(d, src)
        self.lbl_result.setText("Нет данных" if res is None else f"{res[0].date()}: {res[1]}")

# ЛР-4
class EdaTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = EdaConfig()
        self.df: pd.DataFrame | None = None
        self.csv_path: Path | None = None

        lay = QtWidgets.QVBoxLayout(self)

        # выбор файла
        top = QtWidgets.QHBoxLayout()
        self.le_csv = QtWidgets.QLineEdit(); self.le_csv.setReadOnly(False)
        btn_pick = QtWidgets.QPushButton("Выбрать CSV…")
        btn_pick.clicked.connect(self.pick_csv)
        top.addWidget(self.le_csv, 1); top.addWidget(btn_pick)
        lay.addLayout(top)

        # параметры
        grid = QtWidgets.QGridLayout()
        self.sb_threshold = QtWidgets.QDoubleSpinBox()
        self.sb_threshold.setRange(0, 1e9)
        self.sb_threshold.setDecimals(5)       
        self.sb_threshold.setSingleStep(0.01) 
        self.sb_threshold.setValue(0.10)

        self.de_start = QtWidgets.QDateEdit(QtCore.QDate.currentDate().addYears(-1)); self.de_start.setDisplayFormat("yyyy-MM-dd"); self.de_start.setCalendarPopup(True)
        self.de_end = QtWidgets.QDateEdit(QtCore.QDate.currentDate()); self.de_end.setDisplayFormat("yyyy-MM-dd"); self.de_end.setCalendarPopup(True)
        self.sb_year = QtWidgets.QSpinBox(); self.sb_year.setRange(1900, 2100); self.sb_year.setValue(QtCore.QDate.currentDate().year())
        self.sb_month = QtWidgets.QSpinBox(); self.sb_month.setRange(1, 12); self.sb_month.setValue(QtCore.QDate.currentDate().month())

       
        self.btn_prep = QtWidgets.QPushButton("1-4) Загрузить и подготовить")
        self.btn_stats = QtWidgets.QPushButton("5) Статистика + boxplot")
        self.btn_dev = QtWidgets.QPushButton("6) Фильтр по отклонению ≥")
        self.btn_range = QtWidgets.QPushButton("7) Фильтр по диапазону дат")
        self.btn_group = QtWidgets.QPushButton("8) Группировка по месяцам (mean)")
        self.btn_plot_all = QtWidgets.QPushButton("9) График: весь период")
        self.btn_plot_month = QtWidgets.QPushButton("10) График: месяц + median/mean")

        self.btn_prep.clicked.connect(self.load_and_prepare)
        self.btn_stats.clicked.connect(self.show_stats)
        self.btn_group.clicked.connect(self.show_month_group)
        self.btn_dev.clicked.connect(self.apply_deviation_filter)
        self.btn_range.clicked.connect(self.apply_range_filter)
        self.btn_plot_all.clicked.connect(self.plot_all)
        self.btn_plot_month.clicked.connect(self.plot_one_month)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Порог отклонения:"), r, 0); grid.addWidget(self.sb_threshold, r, 1); r += 1
        grid.addWidget(QtWidgets.QLabel("Начальная дата:"), r, 0); grid.addWidget(self.de_start, r, 1)
        grid.addWidget(QtWidgets.QLabel("Конечная дата:"), r, 2); grid.addWidget(self.de_end, r, 3); r += 1
        grid.addWidget(QtWidgets.QLabel("Год:"), r, 0); grid.addWidget(self.sb_year, r, 1)
        grid.addWidget(QtWidgets.QLabel("Месяц:"), r, 2); grid.addWidget(self.sb_month, r, 3); r += 1
        grid.addWidget(self.btn_prep, r, 0); grid.addWidget(self.btn_stats, r, 1); grid.addWidget(self.btn_group, r, 2); r += 1
        grid.addWidget(self.btn_dev, r, 0); grid.addWidget(self.btn_range, r, 1); grid.addWidget(self.btn_plot_all, r, 2); grid.addWidget(self.btn_plot_month, r, 3)
        lay.addLayout(grid)

        # таблица
        self.tbl = QtWidgets.QTableView()
        hdr = self.tbl.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        lay.addWidget(self.tbl)

        # до загрузки блокируем действия
        self._set_actions_enabled(False)

    def _set_actions_enabled(self, on: bool):
        for b in (self.btn_stats, self.btn_group, self.btn_dev, self.btn_range, self.btn_plot_all, self.btn_plot_month):
            b.setEnabled(on)

    def pick_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "CSV с данными", "", "CSV (*.csv)")
        if not path:
            return
        self.csv_path = Path(path)
        self.le_csv.setText(path)
        self._set_actions_enabled(False)  # до успешной загрузки

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "Нет данных", "Сначала загрузите CSV.")
            raise RuntimeError("no df")
        return self.df

    def load_and_prepare(self):
        # путь берем из выбранного файла или из поля
        p = self.csv_path
        if p is None:
            txt = self.le_csv.text().strip()
            if txt:
                p = Path(txt)

        if p is None or not p.exists():
            QtWidgets.QMessageBox.warning(self, "Нет CSV", "Выберите CSV или укажите корректный путь.")
            return

        try:
            df = read_df(p, self.cfg)           # шаги 1–3
            df = add_deviation_cols(df, self.cfg)  # шаг 4
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка при чтении CSV", str(e))
            return

        self.csv_path = p
        self.df = df
        self._show_df(df)
        self._set_actions_enabled(True)

    def show_stats(self):
        df = self._require_df()
        desc, fences = describe_with_outliers(df, self.cfg)
        QtWidgets.QMessageBox.information(self, "Статистика",
            f"{desc.to_string()}\n\nIQR-границы: {fences[0]:.4f} .. {fences[1]:.4f}")

    def apply_deviation_filter(self):
        df = self._require_df()
        thr = float(self.sb_threshold.value())
        self.df = filter_by_deviation(df, thr, self.cfg)
        self._show_df(self.df)

    def apply_range_filter(self):
        df = self._require_df()
        s = pd.Timestamp(self.de_start.date().toString("yyyy-MM-dd"))
        e = pd.Timestamp(self.de_end.date().toString("yyyy-MM-dd"))
        self.df = filter_by_date_range(df, s, e, self.cfg)
        self._show_df(self.df)

    def show_month_group(self):
        df = self._require_df()
        grouped = groupby_month_mean(df, self.cfg)
        self._show_df(grouped)

    def plot_all(self):
        df = self._require_df()
        plot_full_period(df, self.cfg)

    def plot_one_month(self):
        df = self._require_df()
        plot_month(df, int(self.sb_year.value()), int(self.sb_month.value()), self.cfg)

    def _show_df(self, df: pd.DataFrame):
        model = PandasModel(df)
        self.tbl.setModel(model)
        self.tbl.resizeColumnsToContents()

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None): return len(self._df)
    def columnCount(self, parent=None): return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or not index.isValid():
            return None
        val = self._df.iat[index.row(), index.column()]
        if pd.isna(val):
            return ""
        if isinstance(val, (float, int)):
            return f"{val:.5f}"
        return str(val)
    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            col = self._df.columns[section]
            aliases = {
                "date": "date (дата)",
                "rate": "rate (курс тенге→руб)",
                "dev_from_median": "dev_from_median (откл. от медианы)",
                "dev_from_mean": "dev_from_mean (откл. от среднего)",
                "dev_norm": "dev_norm (нормированное откл.)",
            }
            return aliases.get(col, col)
        return section + 1
