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
    result = pd.read_parquet(part / "features_0.parquet")
    assert "snow_depth" not in result.columns
    assert "min_air_temp" not in result.columns
    assert "min_air_temp_min_48h" in result.columns
