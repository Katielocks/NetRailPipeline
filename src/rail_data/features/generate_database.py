import duckdb
import datetime as dt
from pathlib import Path
from typing import Sequence, Union, Any, Iterable
import os
from dateutil import parser
import pandas as pd
import logging

log = logging.getLogger(__name__)


def _sql_value(val: Any) -> str:
    """Return a SQL literal for *val*, preserving numeric types."""
    if isinstance(val, str):
        return f"'{val}'"
    return str(val)


def _iterable_str(values: Iterable[Any]) -> str:
    """Return a `(v1),(v2)…` string for a VALUES clause."""
    return ", ".join(f"({_sql_value(v)})" for v in values)


def generate_main_database(
    loc_ids: Sequence[str | int],
    start_date: dt.datetime,
    end_date: dt.datetime,
    output_dir: Union[str, Path],
    *,
    database: str = ":memory:",
    partition_by: tuple[str, ...] = ("ELR_MIL", "year", "month", "day"),
    write_mode: str = "append",       
    threads: int | None = None,
    memory_limit: str = "12GB",
) -> None:
    """
    Build (or extend) a Hive-partitioned Parquet feature set.

    Parameters
    ----------
    write_mode
        * "append"     → new files alongside the old ones (DuckDB `APPEND`)
        * "overwrite"  → blow away existing partitions (`OVERWRITE`)
        * "ignore"     → skip partitions that already exist (`OVERWRITE_OR_IGNORE`)
    Other arguments are unchanged.
    """

    mode_kw = {
        "append": "APPEND",
        "overwrite": "OVERWRITE",
        "ignore": "OVERWRITE_OR_IGNORE",
    }
    log.info(
        "Generating main feature part from %s to %s", start_date, end_date
    )

    if write_mode not in mode_kw:
        raise ValueError("write_mode must be 'append', 'overwrite', or 'ignore'")

    if threads is None:
        threads = max(1, (os.cpu_count() or 1) // 2)

    loc_values_sql = _iterable_str(loc_ids)

    query = f"""
    WITH locs AS (
        SELECT * FROM (VALUES {loc_values_sql}) AS t(ELR_MIL)
    ),
    params AS (
        SELECT '{start_date}'::timestamp AS p_start,
               '{end_date}'::timestamp   AS p_end
    ),
    hours AS (
        SELECT l.ELR_MIL,
               gs.ts
        FROM locs l
        CROSS JOIN LATERAL generate_series(
            (SELECT p_start FROM params),
            (SELECT p_end   FROM params),
            INTERVAL '1 hour'
        ) AS gs(ts)
    )
    SELECT
        ELR_MIL,
        EXTRACT(year  FROM ts) AS year,
        EXTRACT(month FROM ts) AS month,
        EXTRACT(day   FROM ts) AS day,
        EXTRACT(hour  FROM ts) AS hour,

        sin(2 * pi() * EXTRACT(doy  FROM ts) / 365.25) AS sin_doy,
        cos(2 * pi() * EXTRACT(doy  FROM ts) / 365.25) AS cos_doy,
        sin(2 * pi() * EXTRACT(hour FROM ts) / 24)     AS sin_hod,
        cos(2 * pi() * EXTRACT(hour FROM ts) / 24)     AS cos_hod,

        EXTRACT(dow FROM ts) AS day_of_week
    FROM hours
    """

    # --------------------------- DuckDB session ---------------------------
    con = duckdb.connect(database=database)
    con.execute("INSTALL parquet;")
    con.execute("LOAD parquet;")

    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")

    partition_cols_sql = ", ".join(partition_by)

    copy_stmt = f"""
    COPY (
        {query}
    )
    TO '{Path(output_dir)}'
    (FORMAT PARQUET,
     PARTITION_BY ({partition_cols_sql}),
     {mode_kw[write_mode]});
    """
    con.execute(copy_stmt)
    con.close()
def stream_main_database(
    ELR_MILs: Sequence[str | int],
    start_date: Union[dt.datetime,str] ,
    end_date: Union[dt.datetime,str],
    output_dir: Union[str, Path],
    *,
    database: str = ":memory:",
    window_rule: str | dt.timedelta = "W",
) -> None:
    """
    Generate the feature dataset in rolling windows (weekly, monthly, …),
    which keeps memory use constant for very large date ranges.
    """

    if isinstance(start_date, str):
        start_date = parser.parse(start_date)
    log.info("Streaming main database from %s to %s", start_date, end_date)
    if isinstance(end_date, str):
        end_date = parser.parse(end_date)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    offset = pd.tseries.frequencies.to_offset(window_rule)
    win_start = pd.Timestamp(start_date)

    while win_start <= end_date:
        win_end = win_start + offset - pd.Timedelta(seconds=1)
        win_end = min(win_end, pd.Timestamp(end_date))
        log.debug("Window %s to %s", win_start, win_end)
        generate_main_database(
            ELR_MILs,
            win_start.to_pydatetime(),
            win_end.to_pydatetime(),
            output_dir,
            database=database,
        )

        win_start += offset

    log.info("Finished streaming main database")
