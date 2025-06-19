from __future__ import annotations

import datetime as dt
import logging
import math
import re
from pathlib import Path
from typing import Final, Iterable, Union, Dict, List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import settings, read_cache 
from .utils import location_to_ELR_MIL, sep_datetime  


_YEAR_START: Final[tuple[int, int]] = (4, 1)            # 1 April
_PART_DURATION: Final[dt.timedelta] = dt.timedelta(days=28)
logger = logging.getLogger(__name__)


def _business_year_start(
    when: dt.datetime,
    *,
    year_start: tuple[int, int] = _YEAR_START,
) -> dt.datetime:
    """Return the *start* of the business year that contains ``when``."""
    start_month, start_day = year_start

    dt_year = when.year - 1 if (when.month, when.day) < (start_month, start_day) else when.year
    return dt.datetime(dt_year, start_month, start_day)


def _build_business_period_map(
    start_date: dt.datetime,
    end_date: dt.datetime,
    *,
    year_start: tuple[int, int] = _YEAR_START,
    part_duration: dt.timedelta = _PART_DURATION,
) -> List[str]:
    """List *business‑period* codes that intersect the ``[start_date, end_date]`` window.

    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    start_year_date = _business_year_start(start_date, year_start=year_start)
    end_year_date = _business_year_start(end_date, year_start=year_start)

    periods: List[str] = []

    for year in range(start_year_date.year, end_year_date.year + 1):
        by_start = dt.datetime(year, *year_start)
        by_end = dt.datetime(year + 1, *year_start) - dt.timedelta(days=1)
        code = f"{year}{str(year + 1)[2:]}"  # 2023 + 24 → "202324"

        total_days = (by_end - by_start).days + 1
        parts_per_year = math.ceil(total_days / part_duration.days)

        for part_num in range(1, parts_per_year + 1):
            part_start = by_start + part_duration * (part_num - 1)
            part_end = (
                part_start + part_duration - dt.timedelta(days=1)
                if part_num < parts_per_year
                else by_end
            )
            if part_start <= end_date and part_end >= start_date:
                periods.append(f"{code}_P{part_num:02d}")

    return periods

def _delay_files(
    directory: Path,
    fmt: str,
    *,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
) -> List[Path]:


    pattern = re.compile(fr"^delay_\d{{6}}_[A-Z0-9]{{3}}\.{re.escape(fmt)}$")
    periods = set(_build_business_period_map(start_date, end_date)) if start_date and end_date else None

    matches: List[Path] = []
    for f in directory.iterdir():
        if not f.is_file() or not pattern.match(f.name):
            continue
        if periods and f.stem not in periods:
            continue
        matches.append(f)

    logger.debug("%d delay files matched", len(matches))
    return matches


def _discover_incident_codes(files: Iterable[Path], fmt: str) -> Set[str]:
    """Scan *files* to collect *all* distinct ``INCIDENT_REASON`` codes."""
    codes: Set[str] = set()
    for f in files:
        try:
            df = read_cache(f)
            codes.update(df["INCIDENT_REASON"].dropna().unique())
        except Exception as err:  
            logger.warning("Skipping %s during code discovery: %s", f, err)
    return codes


def extract_incident_dataset(
    *,
    directory: Union[Path, str, None] = None,
    fmt: str | None = None,
    cache_path: Union[Path, str, None] = None,
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    expected_codes: Iterable[str] | None = None,
    scan_codes: bool = True,
) -> None:
   
    if settings and getattr(settings, "delay", None):
        directory = directory or settings.delay.cache
        fmt = fmt or settings.delay.cache_format
    if directory is None or fmt is None:
        raise ValueError("'directory' and 'fmt' must be provided (or via settings)")

    directory = Path(directory)
    cache_path = Path(cache_path or directory / "incident_counts")
    cache_path.mkdir(parents=True, exist_ok=True)

    files = _delay_files(
        directory,
        fmt=fmt,
        start=start_date,
        end=end_date,
    )
    if not files:
        logger.warning("No delay files matched the given criteria – nothing to do.")
        return

    codes: Set[str]
    if expected_codes is not None:
        codes = set(expected_codes)
    elif scan_codes:
        codes = _discover_incident_codes(files, fmt)
    else:
        raise ValueError("'expected_codes' is None and 'scan_codes' is False – "
                         "cannot determine which columns to create.")

    if not codes:
        raise RuntimeError("No INCIDENT_REASON codes could be determined.")

    codes = set(map(str, codes))  
    sorted_codes = sorted(codes)
    sorted_codes = sorted(codes)          # ['FK', 'WF', ...]
    prefix = "INCIDENT_"
    sorted_codes = [f"{prefix}{c}" for c in sorted_codes]
    logger.info("Normalising to %d incident‑reason columns: %s", len(codes), sorted_codes)

    arrow_schema = pa.schema(
        [
            pa.field("EVENT_HOUR", pa.timestamp("ms")),
            pa.field("ELR_MIL", pa.string()),
            *[pa.field(code, pa.int16()) for code in sorted_codes],
        ]
    )

   
    for f in files:
        try:
            df = read_cache(f)
        except Exception as err:
            logger.error("Failed to read %s – skipping. Error: %s", f, err)
            continue

        if not df.empty:
            df = (df.sort_values(["INCIDENT_REASON", "EVENT_DATETIME","ELR_MIL"])
                    .drop_duplicates(subset="INCIDENT_REASON", keep="first"))
            datetime = sep_datetime(df["EVENT_DATETIME"])
            df = df[["ELR_MIL","INCIDENT_REASON"]]
            df = pd.concat([df, datetime], axis=1)

            cols = ["ELR_MIL", "year", "month", "day", "hour"] + ["INCIDENT_REASON"]
            counts = (df
                .groupby(cols)
                .size()                   
                .unstack(fill_value=0)
                .rename(columns=lambda c: f"{prefix}{c}")      
                .sort_index())             

            table = pa.Table.from_pandas(
                counts.reset_index(), schema=arrow_schema, preserve_index=False
            )
            out_file = cache_path / f"{f.stem}.parquet"
            pq.write_table(table, out_file)
            logger.debug("Wrote %s", out_file)

    logger.info("Finished Extracting Incident Dataset: %d files processed", len(files))

