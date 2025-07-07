import sys
import types
import importlib.util
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.features import build_weather_features
from rail_data.features import settings

test_df = pd.DataFrame({
        "ELR_MIL": ["A"],
        "year": [2024],
        "month": [1],
        "day": [1],
        "hour": [0],
        "min_air_temp": [5.0],
        "max_air_temp": [8.0],
        "prcp_amt": [0.1],
        "snow_depth": [0.0],
        "weather_code": ["RA"],
    })

def test_sql_weather_drops_raw_columns(tmp_path: Path):
    part = tmp_path / "ELR_MIL=A" / "year=2024" / "month=1" / "day=1"
    part.mkdir(parents=True)
    test_df.to_parquet(part / "features_0.parquet", index=False)
    build_weather_features(parquet_dir=tmp_path)
    files = list(part.glob("*.parquet"))
    if len(files) == 1:
        result_file = files[0]
    else:
        result_file = next(f for f in files if not f.name.startswith("features_"))
    result = pd.read_parquet(result_file)
    assert "snow_depth" not in result.columns
    assert "min_air_temp" not in result.columns
    assert "min_air_temp_min_48h" in result.columns


def test_sql_weather_builds_raw_when_missing(tmp_path: Path, monkeypatch):
    calls = {}

    def dummy_raw_builder(start_date=None, end_date=None, parquet_dir=None):
        base = Path(parquet_dir or tmp_path)
        part = base / "ELR_MIL=A" / "year=2024" / "month=1" / "day=1"
        part.mkdir(parents=True)
        test_df.to_parquet(part / "features_0.parquet", index=False)
        calls["called"] = True

    monkeypatch.setattr(
        "rail_data.features.sql_weather.build_raw_weather_feature_frame",
        dummy_raw_builder,
    )

    build_weather_features(
        parquet_dir=tmp_path,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-01"),
    )

    part = tmp_path / "ELR_MIL=A" / "year=2024" / "month=1" / "day=1"
    files = list(part.glob("*.parquet"))
    if len(files) == 1:
        result_file = files[0]
    else:
        result_file = next(f for f in files if not f.name.startswith("features_"))
    result = pd.read_parquet(result_file)
    assert calls.get("called", False)
    assert "min_air_temp_min_48h" in result.columns