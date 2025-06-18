from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import duckdb
import pandas as pd
import yaml


AGG_MAP = {
    "min": "MIN",
    "max": "MAX",
    "sum": "SUM",
    "mean": "AVG",
}


_DEF_SETTINGS = Path(__file__).with_name("settings.yaml")


def _load_feature_cfg(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path or _DEF_SETTINGS)
    raw = yaml.safe_load(cfg_path.read_text())
    return raw["weather"]["features"]


def build_weather_features_sql(
    parquet_dir: str | Path,
    cfg_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute weather features using SQL window functions."""

    features = _load_feature_cfg(cfg_path)
    parquet_glob = str(Path(parquet_dir).joinpath("*.parquet"))

    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE weather AS SELECT *, make_timestamp(year, month, day, hour, 0, 0) AS ts FROM parquet_scan('{parquet_glob}')"
    )

    select_parts = ["ELR_MIL", "ts"]
    windows: Dict[str, int] = {}

    for _tbl, cols in features.get("tables", {}).items():
        for col, meta in cols.items():
            func = AGG_MAP.get(meta["action"].lower())
            if not func:
                raise ValueError(f"Unsupported action {meta['action']}")
            hrs = int(meta["window_hours"])
            wname = f"w{hrs}h"
            windows[wname] = hrs
            select_parts.append(
                f"{func}({col}) OVER {wname} AS {col}_{meta['action']}_{hrs}h"
            )

    for flag, meta in features.get("flags", {}).items():
        tbl = next(iter(meta["table"]))
        col_cfg = meta["table"][tbl]
        col = next(iter(col_cfg))
        op = col_cfg[col]["action"].lower()
        hrs = int(col_cfg[col]["window_hours"])
        thresh = meta["threshold"]
        wname = f"w_{flag}"
        windows[wname] = hrs
        cmp = "<=" if op == "le" else ">="
        agg = "MIN" if op == "le" else "MAX"
        select_parts.append(
            f"CASE WHEN {agg}({col}) OVER {wname} {cmp} {thresh} THEN 1 ELSE 0 END AS flag_{flag}"
        )

    window_sql = ", ".join(
        f"{name} AS (PARTITION BY ELR_MIL ORDER BY ts RANGE BETWEEN INTERVAL '{hrs} hours' PRECEDING AND CURRENT ROW)"
        for name, hrs in windows.items()
    )

    query = f"SELECT {', '.join(select_parts)} FROM weather WINDOW {window_sql} ORDER BY ELR_MIL, ts"
    return con.execute(query).df()