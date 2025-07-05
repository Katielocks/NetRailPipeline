

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from ..io import settings as io_settings, get_timetable
from .utils import write_to_parquet, location_to_ELR_MIL, sep_datetime
from .config import settings as feat_settings


_DOW_COLS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _yymmdd_to_datetime(s: pd.Series | pd.Index) -> pd.Series:
    """Convert CIF YYMMDD strings → pandas datetime64[ns]."""
    return pd.to_datetime(s, format="%y%m%d", errors="coerce").dt.normalize()


def _hhmm_to_timedelta(s: pd.Series | pd.Index) -> pd.TimedeltaIndex:
    """Vectorised HHMM → Timedelta (≈6× faster than the string method)."""
    arr = s.to_numpy(dtype=int, copy=False)
    hours = arr // 100
    minutes = arr % 100
    return pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def _explode_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the CIF 7-bit day mask into one row per calendar date.
    """
    mask = df["daysofweek"].to_numpy(dtype=np.uint16)
    bits = ((mask[:, None] >> np.arange(6, -1, -1)) & 1).astype(bool)
    df_bits = pd.DataFrame(bits, columns=_DOW_COLS)

    df = pd.concat([df.reset_index(drop=True), df_bits], axis=1, copy=False)
    out_frames: list[pd.DataFrame] = []

    for dow, col in enumerate(_DOW_COLS):  
        part = df[df[col]]
        if part.empty:
            continue

        start = part["start_date"] + pd.to_timedelta(
            (dow - part["start_date"].dt.weekday) % 7, unit="D"
        )
        counts = ((part["end_date"] - start) // pd.Timedelta(days=7)).astype(int) + 1

        rep_idx = part.index.repeat(counts)
        dates = pd.to_datetime(
            np.hstack(
                [
                    np.arange(s.value, s.value + 7_200_000_000_000 * n, 7_200_000_000_000)
                    for s, n in zip(start, counts, strict=True)
                ]
            )
        ).normalize()

        out_frames.append(part.loc[rep_idx].assign(run_date=dates))

    return pd.concat(out_frames, ignore_index=True)


def _build_hourly_counts(tt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised pipeline → hourly train counts per ELR_MIL.
    No per-row Python loops; complexity O(rows + hours).
    """

    collapsed = (
        tt_df.groupby(
            ["train_id", "ELR_MIL", "start_date", "end_date", "daysofweek"],
            as_index=False,
            sort=False,
        )
        .agg(dep_time=("dep_time", "min"), arr_time=("dep_time", "max"))
    )

    cal = _explode_days(collapsed)

    cal["dep_dt"] = cal["run_date"] + _hhmm_to_timedelta(cal["dep_time"])
    cal["arr_dt"] = cal["run_date"] + _hhmm_to_timedelta(cal["arr_time"])
    cal.loc[cal["arr_dt"] < cal["dep_dt"], "arr_dt"] += pd.Timedelta(days=1)

    dep_h = cal["dep_dt"].values.astype("datetime64[h]").astype("int64")
    arr_h = cal["arr_dt"].values.astype("datetime64[h]").astype("int64")

    locs, loc_idx = np.unique(cal["ELR_MIL"].to_numpy(), return_inverse=True)
    min_h = dep_h.min()
    span = int(arr_h.max() - min_h + 1)

    diff = np.zeros((locs.size, span + 1), dtype=np.int32)
    np.add.at(diff, (loc_idx, dep_h - min_h), 1)
    np.add.at(diff, (loc_idx, arr_h - min_h + 1), -1)
    counts = diff.cumsum(axis=1)[:, :-1]

    run_hours = pd.to_datetime((np.arange(span, dtype=np.int64) + min_h), unit="h")
    counts_df = (
        pd.DataFrame(counts, index=locs, columns=run_hours, copy=False)
        .stack()
        .rename("train_count")
        .reset_index()
        .rename(columns={"level_0": "ELR_MIL", "level_1": "run_hour"})
    )

    parts = sep_datetime(counts_df["run_hour"])
    return pd.concat([counts_df, parts], axis=1, copy=False)



def extract_train_counts(
    *,
    out_root: str | Path | None = None,
    start_date: dt.datetime | str | None = None,
    end_date: dt.datetime | str | None = None,
    partition_cols: Iterable[str] | None = None,
    parquet_compression: str | None = "snappy",
    window_rule: str | dt.timedelta = "W", 
) -> ds.Dataset:
    """
    Stream a large timetable into hourly train-counts, **fast**.

    Parameters
    ----------
    out_root
        Root directory for the Parquet dataset (created if missing).
        Defaults to ``feat_settings.train_counts.parquet_dir``.
    start_date, end_date
        Optional horizon clamp (str | datetime accepted).
    partition_cols
        Column names for Parquet partitioning (order matters).
    parquet_compression
        Codec for Parquet (default ``"snappy"``).
    window_rule
        Size of each processing window.  Examples: ``"ME"`` for months,
        or ``timedelta(days=3)``.
    """

    out_root = (
        Path(out_root)
        if out_root is not None
        else Path(feat_settings.train_counts.parquet_dir).expanduser()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    start_date = pd.to_datetime(start_date).normalize() if start_date else None
    end_date = pd.to_datetime(end_date).normalize() if end_date else None

    if not (io_settings and io_settings.timetable):
        raise RuntimeError("rail_io.io_settings.timetable is required")

    timetable_df = get_timetable(
        io_settings.timetable.cache, start_time=start_date, end_time=end_date
    )[
        ["train_id", "stanox_dep", "dep_time", "start_date", "end_date", "daysofweek"]
    ]
    if timetable_df.empty:
        raise RuntimeError("No timetable rows returned")

    timetable_df["start_date"] = _yymmdd_to_datetime(timetable_df["start_date"])
    timetable_df["end_date"] = _yymmdd_to_datetime(timetable_df["end_date"])
    timetable_df["ELR_MIL"] = location_to_ELR_MIL(timetable_df["stanox_dep"])

    horizon_start = timetable_df["start_date"].min()
    horizon_end = timetable_df["end_date"].max()
    if start_date:
        horizon_start = max(horizon_start, start_date)
    if end_date:
        horizon_end = min(horizon_end, end_date)

    offset = pd.tseries.frequencies.to_offset(window_rule)
    win_start = horizon_start

    while win_start <= horizon_end:
        print(win_start,horizon_end)
        win_end = (win_start + offset) - pd.Timedelta(seconds=1)
        if win_end > horizon_end:
            win_end = horizon_end

        mask = (timetable_df["start_date"] <= win_end) & (
            timetable_df["end_date"] >= win_start
        )
        if not mask.any():
            win_start += offset
            continue

        slice_df = timetable_df.loc[mask].copy()

        slice_df.loc[slice_df["start_date"] < win_start, "start_date"] = win_start
        slice_df.loc[slice_df["end_date"] > win_end, "end_date"] = win_end

        counts = _build_hourly_counts(slice_df)

        write_to_parquet(
            counts,
            out_root,
            partition_cols=partition_cols,
            parquet_compression=parquet_compression,
        )

        win_start += offset

    return ds.dataset(out_root, format="parquet")
