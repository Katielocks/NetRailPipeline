import sys, os
from pathlib import Path
import datetime as dt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.io.delay_extractor import (
    _check_folder,
    _business_year_start,
    _build_business_period_map,
)

def test_check_folder(tmp_path):
    p = tmp_path / "sub"
    p.mkdir()
    f = p / "a.csv"
    f.write_text("x")
    assert _check_folder(p, "csv") is True
    missing = _check_folder(p, "csv", {"b"})
    assert missing == {"b"}
    assert _check_folder(tmp_path / "missing") is False


def test_business_year_start():
    d = dt.datetime(2025, 3, 31)
    start = _business_year_start(d)
    assert start == dt.datetime(2024, 4, 1)


def test_build_business_period_map():
    start = dt.datetime(2024, 4, 1)
    end = dt.datetime(2024, 5, 1)
    periods = _build_business_period_map(start, end)
    assert list(periods) == ["202425"]
    assert periods["202425"] == {"P01", "P02"}