#!/usr/bin/env python
from __future__ import annotations
import math
import logging
import datetime as dt
from pathlib import Path
from typing import Final, MutableMapping, Union, Dict, Set, List
from config import settings

from delay_processer import process_zipfile

_YEAR_START: Final[tuple[int, int]] = (4, 1)  # (month, day) – UK rail FY begins 1 April
_PART_DURATION: Final[dt.timedelta] = dt.timedelta(days=28)

log = logging.getLogger(__name__)

def _check_folder(
    path: Path,
    file_ext: str = None,
    required_parts: Set[str] = None,
) -> Union[bool, List[str]]:
    if not path.is_dir():
        log.debug(f"Directory not found: {path}")
        return False 
    if file_ext:
        matched = list(path.glob(f"*.{file_ext}"))
        if not matched:
            log.debug(f"No .{file_ext} files found in {path}")
            return False
    if required_parts:
        missing: set[str] = set()
        for part in required_parts:
            filename = f"{part}.{file_ext}" if file_ext and not part.endswith(f".{file_ext}") else part
            if not (path / filename).is_file():
                missing.add(filename.split(".")[0])
        if missing:
            log.debug(f"Missing required files in {path}: {', '.join(missing)}")
            return missing
    return True

def _business_year_start(
    dt: dt.datetime,
    *,
    year_start: tuple[int, int] = _YEAR_START,
) -> dt.datetime:
    start_month, start_day = year_start

    if (dt.month, dt.day) < (start_month, start_day):
        dt_year = dt.year - 1
    else:
        dt_year = dt.year

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

    periods: Dict[str, Set[str]] = {}

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
                periods.setdefault(code, set()).add(f"P{part_num:02d}")

    return periods


def _prune_business_period_map(
    business_periods: Dict[str, Set[str]],
    base_dir: Path,
    file_ext: str = None,
    required_parts: Set[str] = None
) -> Dict[str, Set[str]]:
    pruned: Dict[str, Set[str]] = {}

    for code, parts in business_periods.items():
        missing_parts: Set[str] = set()
        year_folder = base_dir / code
        for part in parts:
            part_folder = year_folder / part
            has_part = (
                _check_folder(
                    part_folder,
                    file_ext=file_ext,
                    required_parts=required_parts,
                    error=False
                )
                is True
            )
            if not has_part:
                missing_parts.add(part)
        if missing_parts:
            pruned[code] = missing_parts

    return pruned



def extract_delay_dataset(
    business_period: MutableMapping[str, set[str]] = None,
    overwrite: bool = False,
    src_dir: Path = None,
    out_dir: Path = None,
    out_format: Path = None,

) -> None:
    """Process raw delay ZIPs into cached files.

    Parameters
    ----------
    business_period
        Mapping of business year codes to the set of period identifiers to
        extract.
    overwrite
        If ``True``, overwrite any existing cached files.
    src_dir
        Directory containing the raw ZIP archives downloaded from
        ``raildata.org.uk``.
    out_dir
        Destination directory for processed files.
    out_format
        File extension to use when writing the processed data.
    """
    if not src_dir.is_dir():
        raise SystemExit(f"Input directory {src_dir} does not exist")

    out_dir.mkdir(parents=True, exist_ok=True)

    if business_period is None:
        import_all = True


    raw_delay_check = _check_folder(src_dir, "zip", business_period.keys())
    if raw_delay_check is not True:
        missing_years = ", ".join(str(x) for x in raw_delay_check)
        log.warning(" Missing raw datasets for: %s", missing_years)
        log.info("You can download ZIPs from https://raildata.org.uk/")
        for yr in raw_delay_check:
            business_period.pop(yr, None)
        if not business_period:
            log.info("No remaining datasets – exiting")
            return

    total = 0
    for year in sorted(business_period):
        zip_path = src_dir / f"{year}.zip"
        log.info("Processing %s ...", zip_path.name)
        total += process_zipfile(
            zip_path,
            out_dir,
            out_format,
            overwrite=overwrite,
            business_periods=business_period,
            import_all=import_all,
        )

def get_delay_dataset(
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    src_dir: Path = None,
    out_dir: Path = None,
    out_format: Path = None):
    """Ensure delay datasets exist for the specified date range.

    Parameters
    ----------
    start_date, end_date
        Inclusive date window to cover.  If omitted the entire available range
        is processed.
    src_dir, out_dir
        Override the default input and output directories in ``settings``.
    out_format
        File extension for generated files.

    Returns
    -------
    pandas.DataFrame | None
        ``None`` if all datasets were already present, otherwise the processing
        result from :func:`extract_delay_dataset`.
    """
    if settings and settings.delay:
         src_dir = src_dir or settings.delay.input
         out_dir = out_dir or settings.delay.cache
         out_format = out_format  or settings.delay.cache_format

    business_period_parts: MutableMapping[str, set[str]] = _build_business_period_map(
        start_date,
        end_date,
        _YEAR_START,
    )

    business_period_parts = _prune_business_period_map(business_period_parts, out_dir, "csv", {"delay"})
    if not business_period_parts:
        log.info("Nothing to do - all requested datasets already cached")
        return None
    else:
        extract_delay_dataset(
            business_period=business_period_parts,
            overwrite=False,
            src_dir=src_dir,
            out_dir=out_dir,
            out_format=out_format
            )


