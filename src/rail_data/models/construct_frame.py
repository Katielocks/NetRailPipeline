
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, List, Iterable, Tuple
import glob
import logging
import re
import numpy as np
import duckdb
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_integer_dtype,
)

import statsmodels.formula.api as smf
import statsmodels.api as sm

from ..features.config import settings as feature_settings  # type: ignore

log = logging.getLogger(__name__)

_PARTITION_KEYS = ["ELR_MIL", "year", "month", "day"]    
_JOIN_KEYS      = ["ELR_MIL", "year", "month", "day", "hour"]   

_CONN: duckdb.DuckDBPyConnection | None = None

_BRACE_RE = re.compile(r"\{[^}]*\}")

def _brace(values: Iterable[Union[str, int]]) -> str:
    return "{" + ",".join(str(v) for v in sorted(values)) + "}"



def _has_parquet_files(pattern: str) -> bool:
    """Return True if *pattern* matches at least one .parquet file."""
    safe_pattern = _BRACE_RE.sub("*", pattern)       
    return bool(glob.glob(safe_pattern))

def _get_con() -> duckdb.DuckDBPyConnection:
    global _CONN
    if _CONN is None:
        _CONN = duckdb.connect(database=":memory:")
    return _CONN
    

def _build_glob(
    base: Path,
    elr_mil: str,
    time_filter: Optional[Dict[str, Union[int, List[int]]]],
) -> str:
    parts: list[str] = [f"ELR_MIL={elr_mil}"]
    for key in _PARTITION_KEYS[1:]:
        if time_filter and key in time_filter:
            val = time_filter[key]
            parts.append(f"{key}=" + (_brace(val) if isinstance(val, list) else str(val)))
        else:
            parts.append(f"{key}=*")
    return base.joinpath(*parts).as_posix() + "/*.parquet"


def _first_elr_mil(parquet_dir: str | Path) -> str:
    parquet_dir = Path(parquet_dir)
    for p in sorted(parquet_dir.iterdir()):
        if p.is_dir() and p.name.startswith("ELR_MIL="):
            return p.name.split("=", 1)[1]
    raise FileNotFoundError(f"No ELR_MIL partitions found in {parquet_dir}")


def build_modelling_frame(
    *,
    elr_mil: str | None = None,
    time_filter: Optional[Dict[str, Union[int, List[int]]]] = None,
    incidents_dir: str | Path | None = None,
    weather_dir: str | Path | None = None,
    main_dir: str | Path | None = None,
    timetable_dir: str | Path | None = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge feature tables (INCIDENT, WEATHER, MAIN, TIMETABLE) into one DataFrame.

    • INCIDENT_* columns are left untouched.
    • Any missing table just stays NULL (fills to 0 later).
    """
    incidents_dir  = Path(incidents_dir  or feature_settings.incidents.parquet_dir)
    weather_dir    = Path(weather_dir    or feature_settings.weather.parquet_dir)
    main_dir       = Path(main_dir       or feature_settings.main.parquet_dir)
    timetable_dir  = Path(timetable_dir  or feature_settings.train_counts.parquet_dir)

    if elr_mil is None:
        elr_mil = _first_elr_mil(incidents_dir)
        log.info("Using first ELR_MIL partition: %s", elr_mil)

    proj          = "*" if columns is None else ", ".join(columns)
    main_glob     = _build_glob(main_dir,      elr_mil, time_filter)
    weather_glob  = _build_glob(weather_dir,   elr_mil, time_filter)
    timetable_glob= _build_glob(timetable_dir, elr_mil, time_filter)
    incidents_glob= _build_glob(incidents_dir, elr_mil, time_filter)

    incidents_exist = _has_parquet_files(incidents_glob)

    select_bits = [
        "m.*",
        f"w.* EXCLUDE ({', '.join(_JOIN_KEYS)})",
        f"t.* EXCLUDE ({', '.join(_JOIN_KEYS)})",
    ]
    if incidents_exist:
        select_bits.append(f"i.* EXCLUDE ({', '.join(_JOIN_KEYS)})")
    select_clause = ",\n            ".join(select_bits)

    incidents_cte  = (
        f""",
        incidents AS (
            SELECT {proj} FROM parquet_scan('{incidents_glob}',
                                            hive_partitioning=true,
                                            union_by_name=true)
        )"""
        if incidents_exist else ""
    )
    incidents_join = (
        f"LEFT JOIN incidents i USING ({', '.join(_JOIN_KEYS)})"
        if incidents_exist else ""
    )

    sql = f"""
        WITH
        main AS (
            SELECT {proj} FROM parquet_scan('{main_glob}',
                                            hive_partitioning=true,
                                            union_by_name=true)
        ),
        weather AS (
            SELECT {proj} FROM parquet_scan('{weather_glob}',
                                            hive_partitioning=true,
                                            union_by_name=true)
        ),
        timetable AS (
            SELECT {proj} FROM parquet_scan('{timetable_glob}',
                                            hive_partitioning=true,
                                            union_by_name=true)
        ){incidents_cte}
        SELECT {select_clause}
        FROM main m
        LEFT JOIN weather   w USING ({', '.join(_JOIN_KEYS)})
        LEFT JOIN timetable t USING ({', '.join(_JOIN_KEYS)})
        {incidents_join};
    """

    log.debug("Executing merge query…")
    df = _get_con().sql(sql).df()
    log.info("Loaded %d rows for ELR_MIL=%s", len(df), elr_mil)

    df.drop(columns="__index_level_0__", errors="ignore", inplace=True)
    return df.fillna(0)
