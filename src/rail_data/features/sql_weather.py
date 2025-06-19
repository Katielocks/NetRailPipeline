from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import datetime as dt

import duckdb
import pandas as pd
from .config import settings


AGG_MAP = {
    "min": "MIN",
    "max": "MAX",
    "sum": "SUM",
    "mean": "AVG",
}

def build_weather_features(
    parquet_dir: str | Path,
    *,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    window_rule: str | dt.timedelta = "ME",
) -> None:
    """Streamed variant of :func:`build_weather_features_sql`.

    Parameters
    ----------
    parquet_dir : str | Path
        Location of the raw hourly weather Parquet files.
    start_date, end_date : datetime or None
        Inclusive date range for the output dataset. If None, the earliest
        and latest timestamps across all data will be used.
    window_rule : str | datetime.timedelta, default ``"M"``
        Size of each processing window (e.g. ``"W"`` for weekly).
    """

    parquet_dir = Path(parquet_dir)
    parquet_glob = str(parquet_dir.joinpath("*.parquet"))

    # Determine full date range if not provided
    if start_date is None or end_date is None:
        # Scan all parquet files to get min/max timestamp
        con_stats = duckdb.connect()
        stats_sql = (
            "SELECT MIN(make_timestamp(year,month,day,hour,0,0)), "
            "MAX(make_timestamp(year,month,day,hour,0,0)) "
            f"FROM parquet_scan('{parquet_glob}')"
        )
        min_ts, max_ts = con_stats.execute(stats_sql).fetchone()
        con_stats.close()

        if start_date is None:
            start_date = pd.Timestamp(min_ts)
        if end_date is None:
            end_date = pd.Timestamp(max_ts)

    # Ensure timestamps are pandas Timestamps
    offset = pd.tseries.frequencies.to_offset(window_rule)
    win_start = pd.Timestamp(start_date)
    win_end_limit = pd.Timestamp(end_date)

    while win_start <= win_end_limit:
        win_end = win_start + offset - pd.Timedelta(seconds=1)
        if win_end > win_end_limit:
            win_end = win_end_limit

        con = duckdb.connect()
        con.execute(
            "CREATE OR REPLACE TABLE weather AS "
            f"SELECT *, make_timestamp(year, month, day, hour, 0, 0) AS ts "
            f"FROM parquet_scan('{parquet_glob}') "
            f"WHERE ts BETWEEN '{win_start}' AND '{win_end}'"
        )

        columns = [row[1] for row in con.execute("PRAGMA table_info('weather')").fetchall()]
        select_parts = [c for c in columns if c != "ts"]
        windows: Dict[str, int] = {}

        for _tbl, cols in settings.weather.features.tables.items():
            for col, meta in cols.items():
                func = AGG_MAP.get(meta.action.lower())
                if not func:
                    raise ValueError(f"Unsupported action {meta.action}")
                hrs = int(meta.window_hours)
                wname = f"w{hrs}h"
                windows[wname] = hrs
                select_parts.append(
                    f"{func}({col}) OVER {wname} AS {col}_{meta.action}_{hrs}h"
                )

        for flag, meta in settings.weather.features.flags.items():
            tbl = next(iter(meta.table))
            col_cfg = meta.table[tbl]
            col = next(iter(col_cfg))
            op = col_cfg[col].action.lower()
            hrs = int(col_cfg[col].window_hours)
            thresh = meta.threshold
            wname = f"w_{flag}"
            windows[wname] = hrs
            cmp_op = "<=" if op == "le" else ">="
            agg = "MIN" if op == "le" else "MAX"
            select_parts.append(
                f"CASE WHEN {agg}({col}) OVER {wname} {cmp_op} {thresh} "
                f"THEN 1 ELSE 0 END AS flag_{flag}"
            )

        window_sql = ", ".join(
            f"{name} AS (PARTITION BY ELR_MIL ORDER BY ts "
            f"RANGE BETWEEN INTERVAL '{hrs} hours' PRECEDING AND CURRENT ROW)"
            for name, hrs in windows.items()
        )

        query = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM weather WINDOW {window_sql} "
            f"ORDER BY ELR_MIL, ts"
        )

        df = con.execute(query).fetchdf()
        df.to_parquet(
            parquet_dir,
            partition_cols=["ELR_MIL", "year", "month", "day", "hour"],
            engine="pyarrow",
            index=False,
        )

        con.close()
        win_start += offset
