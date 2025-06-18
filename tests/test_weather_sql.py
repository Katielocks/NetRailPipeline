import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rail_data.features.sql_weather import build_weather_features_sql


def test_build_weather_features_sql(tmp_path):
    df = pd.DataFrame(
        {
            "ELR_MIL": ["A"] * 4,
            "year": [2024] * 4,
            "month": [4] * 4,
            "day": [1] * 4,
            "hour": [0, 1, 2, 3],
            "min_air_temp": [5, 4, 3, 2],
            "max_air_temp": [10, 11, 12, 13],
            "prcp_amt": [0.0, 1.0, 0.5, 0.2],
            "SNOW_DEPTH": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(df), tmp_path / "data.parquet")

    result = build_weather_features_sql(tmp_path)
    assert "prcp_amt_sum_6h" in result.columns
    assert "flag_freeze" in result.columns