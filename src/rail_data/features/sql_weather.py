from __future__ import annotations


from pathlib import Path
from typing import Dict, List
import datetime as dt
import logging
import tempfile
from contextlib import nullcontext

import duckdb
import pandas as pd

from .config import settings
from .convert_weather import build_raw_weather_feature_frame
from .utils import write_to_parquet

log = logging.getLogger(__name__)

AGG_MAP: Dict[str, str] = {"min": "MIN", "max": "MAX", "sum": "SUM", "mean": "AVG"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _mk_parquet_expr(base: Path) -> str:
    """Return a DuckDB parquet_scan expression for *base* (file or dir).

    When *base* is a directory we try to discover the most recent raw
    parquet partition for each station and return a list expression that
    DuckDB can expand.
    """
    if base.is_file():
        return f"['{base}']"

    def _raw_candidates(dir_: Path) -> List[Path]:
        return [p for p in dir_.glob("*.parquet") if not p.name.startswith("features_")]

    def _any_candidates(dir_: Path) -> List[Path]:
        return list(dir_.glob("*.parquet"))

    files: List[str] = []
    for d in sorted(base.glob("ELR_MIL=*/year=*/month=*/day=*")):
        hourly = sorted(_raw_candidates(d))
        if not hourly:
            hourly = sorted(_any_candidates(d))
        if hourly:
            files.append(str(hourly[0]))

    if not files:
        files = [str(p) for p in sorted(_raw_candidates(base))]
        if not files:
            files = [str(p) for p in sorted(base.glob("*.parquet"))]

    if not files:
        raise FileNotFoundError(f"No raw Parquet files under {base}")

    return "[" + ", ".join(f"'{p}'" for p in files) + "]"


def _max_window_hours() -> int:
    """Return the largest rolling‑window span (hours) configured."""
    max_h = 0
    for _tbl, col_map in settings.weather.features.tables.items():
        for _col, meta in col_map.items():
            max_h = max(max_h, int(meta.window_hours))
    for _flag, flag_cfg in settings.weather.features.flags.items():
        tbl = next(iter(flag_cfg.table))
        col_cfg = flag_cfg.table[tbl]
        col = next(iter(col_cfg))
        max_h = max(max_h, int(col_cfg[col].window_hours))
    return max_h


def _drop_old_raw_partitions(base: Path, keep_after: dt.datetime) -> None:
    """Delete raw partitions whose date is strictly earlier than *keep_after*."""
    keep_date = keep_after.date()
    for p in base.glob("ELR_MIL=*/year=*/month=*/day="):
        try:
            day = int(p.name.split("=")[1])
            month = int(p.parent.name.split("=")[1])
            year = int(p.parent.parent.name.split("=")[1])
        except (IndexError, ValueError):
            continue
        if dt.date(year, month, day) < keep_date:
            for f in p.glob("*.parquet"):
                f.unlink(missing_ok=True)
            try:
                p.rmdir()
            except OSError:
                pass



def build_weather_features(
    *,
    parquet_dir: str | Path | None = None,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    window_rule: str | dt.timedelta = "W",
    build_raw: bool = True,
) -> None:
    """Generate feature parquet partitions for the requested date range.

    Parameters
    ----------
    parquet_dir
        Root directory where *feature* parquet files (not raw!) are
        written.  Defaults to ``settings.weather.parquet_dir``.
    start_date, end_date
        Inclusive UTC bounds of the time window to materialise.  When
        omitted, the bounds are inferred from whatever raw data are
        already present.
    window_rule
        Pandas offset alias or ``timedelta`` describing how the overall
        time span is chunked.  Each chunk is processed independently so
        memory usage stays bounded.
    build_raw
        When *True* (default) the function ensures the required *raw*
        hourly parquet exists, secretly creating missing bits in a temp
        directory if needed.  When *False* we assume callers prepared
        raw parquet themselves.
    """

    log.info("Building weather features from %s to %s", start_date, end_date)


    if settings and settings.weather:
        parquet_dir = parquet_dir or settings.weather.parquet_dir

    feature_base = Path(parquet_dir).expanduser().resolve()
    feature_base.mkdir(parents=True, exist_ok=True)

    try:
        _ = _mk_parquet_expr(feature_base)
        need_temp_raw = build_raw is True and False  
    except FileNotFoundError:
        need_temp_raw = build_raw 

    raw_ctx = (
        tempfile.TemporaryDirectory() if need_temp_raw else nullcontext(str(feature_base))
    )
    max_buffer = pd.Timedelta(hours=_max_window_hours())

    with raw_ctx as raw_dir_str:
        raw_base = Path(raw_dir_str).resolve()

        try:
            parquet_expr = _mk_parquet_expr(raw_base)
            with duckdb.connect() as con:
                (_min_ts, last_raw_end) = con.execute(
                    "SELECT MIN(make_timestamp(year,month,day,hour,0,0)), "
                    "       MAX(make_timestamp(year,month,day,hour,0,0)) "
                    f"FROM parquet_scan({parquet_expr}, HIVE_PARTITIONING=1)"
                ).fetchone()
                last_raw_end = pd.Timestamp(last_raw_end) if last_raw_end else None
        except FileNotFoundError:
            if start_date is None or end_date is None:
                raise  
            init_start = pd.Timestamp(start_date) - max_buffer

            build_raw_weather_feature_frame(
                start_date=init_start,
                end_date=end_date,
                parquet_dir=raw_base,
            )
            last_raw_end = pd.Timestamp(end_date)
            parquet_expr = _mk_parquet_expr(raw_base)

        if start_date is None or end_date is None:
            with duckdb.connect() as con:
                min_ts, max_ts = con.execute(
                    "SELECT MIN(make_timestamp(year,month,day,hour,0,0)), "
                    "       MAX(make_timestamp(year,month,day,hour,0,0)) "
                    f"FROM parquet_scan({parquet_expr}, HIVE_PARTITIONING=1)"
                ).fetchone()
            start_date = start_date or pd.Timestamp(min_ts)
            end_date = end_date or pd.Timestamp(max_ts)


        offset = pd.tseries.frequencies.to_offset(window_rule)
        win_start = pd.Timestamp(start_date)
        win_end_lim = pd.Timestamp(end_date)

        while win_start <= win_end_lim:
            part_start = win_start - max_buffer
            win_end = min(win_end_lim, win_start + offset - pd.Timedelta(seconds=1))

            if build_raw:
                build_start = part_start if last_raw_end is None else max(
                    part_start, last_raw_end + pd.Timedelta(hours=1)
                )
                if last_raw_end is None or build_start <= win_end:
                    build_raw_weather_feature_frame(
                        start_date=build_start,
                        end_date=win_end,
                        parquet_dir=raw_base,
                    )
                    parquet_expr = _mk_parquet_expr(raw_base)
                    last_raw_end = win_end

            log.debug("Weather window %s → %s", win_start, win_end)


            with duckdb.connect() as con:
                con.execute(
                    f"""
                    CREATE OR REPLACE TABLE weather AS
                    SELECT *, make_timestamp(year, month, day, hour, 0, 0) AS ts
                    FROM parquet_scan({parquet_expr}, HIVE_PARTITIONING=1, UNION_BY_NAME=1)
                    WHERE ts BETWEEN '{part_start}' AND '{win_end}'
                    """
                )

                cols = [r[1] for r in con.execute("PRAGMA table_info('weather')").fetchall()]
                drop_cols: set[str] = set()
                for _tbl, col_map in settings.weather.features.tables.items():
                    drop_cols.update(col_map.keys())
                for _flag, flag_cfg in settings.weather.features.flags.items():
                    tbl = next(iter(flag_cfg.table))
                    col_cfg = flag_cfg.table[tbl]
                    drop_cols.add(next(iter(col_cfg)))

                raw_cols = [c for c in cols if c not in {"ts"} and c not in drop_cols]

                feat_terms: List[str] = []
                agg_names: List[str] = []
                windows: Dict[str, int] = {}
                for _tbl, col_map in settings.weather.features.tables.items():
                    for col, meta in col_map.items():
                        func = AGG_MAP[meta.action.lower()]
                        hrs = int(meta.window_hours)
                        wname = f"w{hrs}h"
                        windows[wname] = hrs
                        new_name = f"{col}_{meta.action}_{hrs}h"
                        agg_names.append(new_name)
                        feat_terms.append(f"{func}({col}) OVER {wname} AS {new_name}")

                for flag, meta in settings.weather.features.flags.items():
                    tbl = next(iter(meta.table))
                    col_cfg = meta.table[tbl]
                    col = next(iter(col_cfg))
                    op = col_cfg[col].action.lower()
                    hrs = int(col_cfg[col].window_hours)
                    thresh = meta.threshold
                    wname = f"w_{flag}"
                    windows[wname] = hrs
                    cmp, agg = ("<=", "MIN") if op == "le" else (">=", "MAX")
                    new_name = f"flag_{flag}"
                    agg_names.append(new_name)
                    feat_terms.append(
                        f"CASE WHEN {agg}({col}) OVER {wname} {cmp} {thresh} THEN 1 ELSE 0 END AS {new_name}"
                    )

                window_sql = ", ".join(
                    f"{n} AS (PARTITION BY ELR_MIL ORDER BY ts RANGE BETWEEN INTERVAL '{h} hours' PRECEDING AND CURRENT ROW)"
                    for n, h in windows.items()
                )

                df_feat = con.execute(
                    "SELECT "             
                    + ", ".join(["ts"] + raw_cols)          
                    + (", " + ", ".join(feat_terms) if feat_terms else "")
                    + " FROM weather WINDOW "
                    + window_sql
                    + " ORDER BY ELR_MIL, ts"
                ).fetchdf()

            df_feat[agg_names] = df_feat.groupby("ELR_MIL", sort=False)[agg_names].transform(lambda x: x.bfill())
            df_feat = df_feat[df_feat["ts"].between(win_start, win_end)].copy()

            df_feat = df_feat.drop(columns="ts")
            log.info("Writing parquet to %s from %s to %s", feature_base, start_date, end_date)
    
            write_to_parquet(
                df_feat,
                out_root=feature_base,
                partition_cols=["ELR_MIL", "year", "month", "day"],
            )

            if build_raw and raw_base != feature_base:
                keep_after = win_start + offset - max_buffer
                _drop_old_raw_partitions(raw_base, keep_after)

            win_start += offset

    log.info("Weather feature generation complete")
