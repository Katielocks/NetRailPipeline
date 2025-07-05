from __future__ import annotations

from pathlib import Path
from typing import Dict, List
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


def _hive_part_dir(base: Path, elr: str, y: int, m: int, d: int) -> Path:
    return (
        base
        / f"ELR_MIL={elr}"
        / f"year={y}"
        / f"month={m}"
        / f"day={d}"
    )


def _clean_partitions(base: Path, df: pd.DataFrame) -> None:
    for elr, y, m, d in df[["ELR_MIL", "year", "month", "day"]].drop_duplicates().itertuples(index=False):
        p = _hive_part_dir(base, elr, y, m, d)
        if p.exists():
            for f in p.glob("*.parquet"):
                f.unlink()
        else:
            p.mkdir(parents=True, exist_ok=True)


def _write_feature_files(base: Path, df: pd.DataFrame) -> None:
    for (elr, y, m, d), g in df.groupby(["ELR_MIL", "year", "month", "day"], sort=False):
        p = _hive_part_dir(base, elr, y, m, d)
        p.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["ELR_MIL", "year", "month", "day"]).to_parquet(p / "features_0.parquet", index=False, engine="pyarrow")


def _mk_parquet_expr(base: Path) -> str:
    if base.is_file():
        return f"['{base}']"
    files: List[str] = []
    for d in sorted(base.glob("ELR_MIL=*/year=*/month=*/day=*")):
        hourly = sorted(d.glob("*.parquet"))
        if hourly:
            files.append(str(hourly[0]))
    if not files:
        files = [str(p) for p in sorted(base.glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"No Parquet files under {base}")
    return "[" + ", ".join(f"'{p}'" for p in files) + "]"


def build_weather_features(
    *,
    parquet_dir: str | Path | None = None,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    window_rule: str | dt.timedelta = "W"
) -> None:
    """Create feature Parquets retaining all raw columns & adding aggregates/flags.
    """

    if settings and settings.weather:
        parquet_dir = parquet_dir or settings.weather.parquet_dir
    base = Path(parquet_dir).expanduser().resolve()
    parquet_expr = _mk_parquet_expr(base)


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
        win_end = min(win_end_lim, win_start + offset - pd.Timedelta(seconds=1))

        with duckdb.connect() as con:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE weather AS
                SELECT *, make_timestamp(year, month, day, hour, 0, 0) AS ts
                FROM parquet_scan({parquet_expr}, HIVE_PARTITIONING=1, UNION_BY_NAME=1)
                WHERE ts BETWEEN '{win_start}' AND '{win_end}'
                """
            )


            cols = [r[1] for r in con.execute("PRAGMA table_info('weather')").fetchall()]
            drop_cols = set()
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
                feat_terms.append(f"CASE WHEN {agg}({col}) OVER {wname} {cmp} {thresh} THEN 1 ELSE 0 END AS {new_name}")

            window_sql = ", ".join(
                f"{n} AS (PARTITION BY ELR_MIL ORDER BY ts RANGE BETWEEN INTERVAL '{h} hours' PRECEDING AND CURRENT ROW)"
                for n, h in windows.items()
            )

            select_clause = (
                "SELECT "
                + ", ".join(raw_cols) 
                + (", " + ", ".join(feat_terms) if feat_terms else "")
                + " FROM weather WINDOW "
                + window_sql
                + " ORDER BY ELR_MIL, ts"
            )

            df_feat = con.execute(select_clause).fetchdf()

        df_feat[agg_names] = df_feat.groupby("ELR_MIL", sort=False)[agg_names].transform(lambda x: x.bfill())

        _clean_partitions(base, df_feat)
        _write_feature_files(base, df_feat)

        win_start += offset
