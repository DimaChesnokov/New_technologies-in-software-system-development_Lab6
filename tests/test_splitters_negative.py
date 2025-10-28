from pathlib import Path
from src.splitters import split_by_year, split_by_week

def test_split_by_year_empty_returns_empty_list(tmp_path: Path):
    p = tmp_path/"empty.csv"
    p.write_text("date,value\n", encoding="utf-8")
    out = tmp_path/"by_year"; out.mkdir()
    assert split_by_year(p, out) == []

def test_split_by_week_empty_returns_empty_list(tmp_path: Path):
    p = tmp_path/"empty.csv"
    p.write_text("date,value\n", encoding="utf-8")
    out = tmp_path/"by_week"; out.mkdir()
    assert split_by_week(p, out) == []
