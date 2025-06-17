import pandas as pd
import math
from typing import Union,Final,Dict,List,Set,Iterable
from pathlib import Path
import datetime as dt
import re
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import logging

from ..rail_io import settings,read_cache
from .utils import location_to_ELR_MIL,write_to_parquet,sep_datetime

_YEAR_START: Final[tuple[int, int]] = (4, 1) 
_PART_DURATION: Final[dt.timedelta] = dt.timedelta(days=28)

logger = logging.getLogger(__name__)

def _business_year_start(
    when: dt.datetime,
    *,
    year_start: tuple[int, int] = _YEAR_START) -> dt.datetime:
    """Return the start of the business year for ``when``."""
    start_month, start_day = year_start

    if (when.month, when.day) < (start_month, start_day):
        dt_year = when.year - 1
    else:
        dt_year = when.year

    return dt.datetime(dt_year, start_month, start_day)


def _build_business_period_map(
    start_date: dt.datetime,
    end_date: dt.datetime,
    *,
    year_start: tuple[int, int] = _YEAR_START,
    part_duration: dt.timedelta = _PART_DURATION,
) -> Dict[str, Set[str]]:
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    start_year_date = _business_year_start(start_date, year_start=year_start)
    end_year_date = _business_year_start(end_date, year_start=year_start)

    periods: List[str] = {}

    for year in range(start_year_date.year, end_year_date.year + 1):
        by_start = dt.datetime(year, *year_start)
        by_end = dt.datetime(year + 1, *year_start) - dt.timedelta(days=1)
        code = f"{year}{str(year + 1)[2:]}" 
        total_days = (by_end - by_start).days + 1
        parts_per_year = math.ceil(total_days / part_duration.days)
        for part_num in range(1, parts_per_year + 1):
            part_start = by_start + part_duration * (part_num - 1)
            if part_num < parts_per_year:
                part_end = part_start + part_duration - dt.timedelta(days=1)
            else:
                part_end = by_end
            if part_start <= end_date and part_end >= start_date:
                periods.append(f"{code}_P{part_num:02d}")

    return periods


def _delay_files(
    directory: Path,
    fmt: str,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[Path]:
    """
    Return delay data files whose business periods intersect [start, end].
    """
    pattern = re.compile(fr"^delay_\d{{6}}_[A-Z0-9]{{3}}\.{re.escape(fmt)}$")
    periods = (
        set(_build_business_period_map(start, end)) if start and end else None
    )

    matches: list[Path] = []
    for f in directory.iterdir():
        if not f.is_file() or not pattern.match(f.name):
            continue
        if periods and f.stem not in periods:
            continue
        matches.append(f)
    return matches




def extract_incident_dataset(directory: Union[Path|str] = None, 
                          fmt: str = None, 
                          cache_path: Union[Path|str] = None, 
                          start_date: dt.datetime= None, 
                          end_date: dt.datetime= None):
    if settings and settings.delay:
        directory = directory or settings.delay.cache_dir
        fmt = fmt or settings.delay.cache_format
    files = _delay_files(directory,start_date=start_date,end_date=end_date,fmt=fmt)
    for file in files:
        df = read_cache(f"{directory}/{file}.{fmt}")
        df["ELR_MIL"] = location_to_ELR_MIL(df["SECTION_CODE"].str.split(':')[0])
        df = df[["EVENT_DATETIME","INCIDENT_REASON","ELR_MIL"]]
        if not df.empty:
            df = (df.sort_values(["INCIDENT_REASON", "EVENT_DATETIME","ELR_MIL"])
                    .drop_duplicates(subset="INCIDENT_REASON", keep="first"))
            datetime = sep_datetime(df["EVENT_DATETIME"])
            df = df["ELR_MIL","INCIDENT_REASON"]
            df = pd.concat([df, datetime], axis=1)
            write_to_parquet(df,cache_path)
    
            
