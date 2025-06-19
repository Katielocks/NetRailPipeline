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
        "SNOW_DEPTH": [0.0],
        "weather_code": ["RA"],
    })

def test_sql_weather_keeps_columns(tmp_path: Path):
    test_df.to_parquet(tmp_path / "input.parquet", index=False)

    build_weather_features(tmp_path)
    (tmp_path / "input.parquet").unlink()
    result = pd.read_parquet(tmp_path)
    assert "weather_code" in result.columns
    assert "SNOW_DEPTH" in result.columns
