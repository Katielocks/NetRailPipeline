from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from ..io import settings, get_timetable
from .utils import write_to_parquet,location_to_ELR_MIL,sep_datetime


def _yymmdd_to_datetime(s: pd.Series | pd.Index) -> pd.Series:
    """Convert CIF YYMMDD strings → pandas datetime64[ns]."""
    return pd.to_datetime(s, format="%y%m%d", errors="coerce")


def _hhmm_to_timedelta(s: pd.Series | pd.Index) -> pd.Series:
    """Convert HHMM (string/int) → pandas Timedelta."""
    s = s.astype(str).str.zfill(4)
    return (
        pd.to_timedelta(s.str.slice(0, 2).astype(int), unit="h")
        + pd.to_timedelta(s.str.slice(2, 4).astype(int), unit="m")
    )


def _explode_days(
    df: pd.DataFrame,
    mask_col: str = "daysofweek",
    start_col: str = "start_date",
    end_col: str = "end_date",
) -> pd.DataFrame:
    """Vectorised explosion of CIF bit-mask to calendar dates per row."""

    bits = (
        df[mask_col]
        .astype(str)
        .str.zfill(7)
        .apply(list)
        .apply(lambda lst: list(map(int, lst)))
    )
    bits_df = pd.DataFrame(bits.tolist(), columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    df = pd.concat([df.reset_index(drop=True), bits_df], axis=1)

    def _calendar_row(row) -> pd.DatetimeIndex:
        alldays = pd.date_range(row[start_col], row[end_col], freq="D")
        mask = np.array(
            [row["Mon"], row["Tue"], row["Wed"], row["Thu"], row["Fri"], row["Sat"], row["Sun"]],
            dtype=bool,
        )
        return alldays[np.isin(alldays.weekday, np.flatnonzero(mask))]

    calendars = df.apply(_calendar_row, axis=1)
    return df.loc[df.index.repeat(calendars.str.len())].assign(run_date=np.concatenate(calendars.values))


def _build_hourly_counts(tt_df: pd.DataFrame, geo_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pipeline that returns hourly counts.  If *geo_df* is supplied we
    perform the STANOX to ELR_MIL merge; otherwise we assume the column is
    already present in *tt_df*.
    """

    if geo_df is not None:
        tt_df = tt_df.merge(
            geo_df[["STANOX", "ELR_MIL"]],
            left_on="stanox_dep",
            right_on="STANOX",
            how="left",
        )
    elif "ELR_MIL" not in tt_df.columns:
        raise ValueError("ELR_MIL column missing and no geo_df provided")

    collapsed = (
        tt_df.groupby(["train_id", "ELR_MIL", "start_date", "end_date", "daysofweek"], as_index=False)
        .agg(dep_time=("dep_time", "min"), arr_time=("dep_time", "max"))
    )

    cal = _explode_days(collapsed)

    cal["dep_dt"] = cal["run_date"] + _hhmm_to_timedelta(cal["dep_time"])
    cal["arr_dt"] = cal["run_date"] + _hhmm_to_timedelta(cal["arr_time"])
    cal.loc[cal["arr_dt"] < cal["dep_dt"], "arr_dt"] += pd.Timedelta(days=1)

    hours = (
        cal.loc[:, ["ELR_MIL"]]
        .join(
            cal.apply(
                lambda row: pd.date_range(
                    row["dep_dt"].floor("h"),
                    row["arr_dt"].floor("h"),
                    freq="h",
                ),
                axis=1,
            ).explode()
        )
    )
    hours.columns = ["ELR_MIL", "run_hour"]

    counts = hours.value_counts(["ELR_MIL", "run_hour"]).rename("train_count").reset_index()

    datetime_parts = sep_datetime(counts['run_hour'])
    counts = pd.concat([counts, datetime_parts], axis=1)
    
    return counts


def extract_train_counts(
    *,
    out_root: str | Path,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    partition_cols: Iterable[str] = None,
    parquet_compression: str | None = "snappy",
    window_rule: str | dt.timedelta = "W" 
) -> ds.Dataset:
    """Stream a large timetable into hourly train-counts.

    Parameters
    ----------
    out_root : str | Path
        Root directory for the Parquet dataset (created if missing).
    partition_cols : Iterable[str], default see module constant
        Column names for partitioning.  **Order matters**.
    parquet_compression : str | None, default "snappy"
        Codec for :pyarrow:`write_to_dataset`.
    window_rule : str | datetime.timedelta, default "W"
        Size of each processing window.  Examples: ``"M"`` for calendar
        months, or ``timedelta(days=3)``.
    """

    if not (settings and settings.timetable and settings.geospatial):
        raise RuntimeError(
            "rail_io.settings is incomplete – timetable and geospatial paths required"
        )

    out_path = Path(out_root).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    timetable_df = get_timetable(settings.timetable.cache)[
        [
            "train_id",
            "stanox_dep",
            "dep_time",
            "arr_time",
            "start_date",
            "end_date",
            "daysofweek",
        ]
    ]
    if timetable_df.empty:
        raise RuntimeError("No timetable rows returned")

    timetable_df["start_date"] = _yymmdd_to_datetime(timetable_df["start_date"]).dt.normalize()
    timetable_df["end_date"] = _yymmdd_to_datetime(timetable_df["end_date"]).dt.normalize()



    horizon_start = timetable_df["start_date"].min()
    horizon_end = timetable_df["end_date"].max()

    if start_date and horizon_start < start_date:
        horizon_start = start_date
    if end_date and horizon_end > end_date:
        horizon_end = end_date


    offset = pd.tseries.frequencies.to_offset(window_rule)
    win_start = horizon_start

    while win_start <= horizon_end:
        win_end = win_start + offset - pd.Timedelta(seconds=1)
        if win_end > horizon_end:
            win_end = horizon_end

        filter_start = win_start
        filter_end = win_end

        slice_mask = (timetable_df["start_date"] <= filter_end) & (
            timetable_df["end_date"] >= filter_start
        )
        if not slice_mask.any():
            win_start += offset
            continue  

        slice_df = timetable_df.loc[slice_mask].copy()

        slice_df.loc[slice_df["start_date"] < win_start, "start_date"] = win_start
        slice_df.loc[slice_df["end_date"] > win_end, "end_date"] = win_end

        slice_df["ELR_MIL"] = location_to_ELR_MIL(slice_df["stanox_dep"])

        counts = _build_hourly_counts(slice_df, geo_df=None)

        write_to_parquet(
            counts,
            out_root,
            partition_cols,
            parquet_compression
        )

    return ds.dataset(out_path, format="parquet")
