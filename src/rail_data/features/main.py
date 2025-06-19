from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..io import settings as io_settings
from .convert_weather import build_raw_weather_feature_frame
from .sql_weather import build_weather_features_sql
from .streaming_train_counts import extract_train_counts
from .extract_incidents import extract_incident_dataset
from .generate_database import generate_main_database
from .config import settings


def _as_datetime(val: dt.date | dt.datetime | str) -> dt.datetime:
    if isinstance(val, dt.datetime):
        return val
    if isinstance(val, dt.date):
        return dt.datetime.combine(val, dt.time.min)
    if isinstance(val, str):
        return dt.datetime.fromisoformat(val)
    raise TypeError(f"Unsupported date type: {type(val)!r}")


def create_datasets(start_date: dt.date | dt.datetime | str,
                    end_date: dt.date | dt.datetime | str) -> None:
    """Generate feature datasets between ``start_date`` and ``end_date``.

    Existing Parquet data will be overwritten.
    """
    start_dt = _as_datetime(start_date)
    end_dt = _as_datetime(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    
    generate_main_database(start_date,end_date)

    build_raw_weather_feature_frame(start_date=start_dt, end_date=end_dt)
    build_weather_features_sql(parquet_dir=settings.weather.parquet_dir)

    extract_train_counts(out_root=settings.train_counts.parquet_dir,start_date=start_date,end_date=end_date)

    extract_incident_dataset(
        directory=Path(io_settings.delay.cache),
        fmt=io_settings.delay.cache_format,
        cache_path=settings.incidents.parquet_dir,
        start_date=start_dt,
        end_date=end_dt,
    )