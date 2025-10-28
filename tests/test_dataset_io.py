from pathlib import Path
import pandas as pd
from src.dataset_io import save_XY, read_two_columns
from src.config import DATE_FMT

def test_save_XY_and_read(tmp_path: Path):
    # временный CSV
    input_csv = tmp_path / "data.csv"
    input_csv.write_text("date,value\n2024-10-01,0.20\n2024-10-02,0.25\n", encoding="utf-8")

    x_csv = tmp_path / "X.csv"
    y_csv = tmp_path / "Y.csv"

    # сохраняем X и Y
    save_XY(input_csv, x_csv, y_csv)

    #  X
    x_df = pd.read_csv(x_csv)
    assert list(x_df.columns) == ["date"]
    assert x_df["date"].iloc[0] == "2024-10-01"

    #  Y
    y_df = pd.read_csv(y_csv)
    assert list(y_df.columns) == ["value"]
    assert abs(y_df["value"].iloc[1] - 0.25) < 1e-6

def test_read_two_columns_sorted(tmp_path: Path):
    p = tmp_path / "unsorted.csv"
    # неотсортированные даты
    p.write_text("date,value\n2024-10-03,0.22\n2024-10-01,0.20\n2024-10-02,0.21\n", encoding="utf-8")

    df = read_two_columns(p)
    # отсортировано
    assert list(df["date"].dt.strftime(DATE_FMT)) == ["2024-10-01","2024-10-02","2024-10-03"]
    assert df["value"].iloc[0] == 0.20
