from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

from .config import settings
from .utils import write_cache,get_cache
from .cif_hop_extract import extract_CIF

log = logging.getLogger(__name__)


def _collect_zip_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("*.zip"))

def _get_timetable_periods(start, end):
    first_by = start.year if start.month >= 5 else start.year - 1
    last_by = end.year if end.month >= 5 else end.year - 1

    periods = []
    for by in range(first_by, last_by + 1):
        may_start = dt.datetime(by, 5, 1)
        may_end = dt.datetime(by, 11, 30, 23, 59, 59)
        code_may = f"{by}{str(by + 1)[-2:]}_MAY"
        periods.append((code_may, may_start, may_end))

        dec_start = dt.datetime(by, 12, 1)
        dec_end = dt.datetime(by + 1, 4, 30, 23, 59, 59)
        code_dec = f"{by}{str(by + 1)[-2:]}_DEC"
        periods.append((code_dec, dec_start, dec_end))

    filtered = [
        (code, p_start, p_end) for code, p_start, p_end in periods
        if not (p_end < start or p_start > end)
    ]

    filtered.sort(key=lambda x: x[1])
    codes = [code for code, _, _ in filtered]
    return codes


def extract_timetable(
    start_time: dt.datetime = None,
    end_time: dt.datetime = None,
    *,
    input_path: Union[str, Path] = None,
    cache_path: Union[str, Path] | None = None,
) -> pd.DataFrame:
    """Consolidate CIF timetable zips into a single **hops** :class:`~pandas.DataFrame`.

    Parameters
    ----------
    input_path
        Directory containing Network Rail CIF ZIP archives *or* a single ZIP
        file.
    cache_path
        If set, the resulting DataFrame is cached to *cache_path*.
    start_time, end_time
        Inclusive date window.  If provided, only timetable periods
        intersecting this window will be processed.  **Both** parameters must
        be supplied together, and ``start_time`` must not be later than
        ``end_time``.

    Returns
    -------
    pandas.DataFrame
        Concatenated hops extracted from the selected CIF archives.
    """
    if settings and settings.timetable:
        input_path = input_path or settings.timetable.input
        cache_path = cache_path or settings.timetable.cache
        
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if (start_time is None) ^ (end_time is None):
        raise ValueError("Provide *both* 'start_time' and 'end_time' to filter by date window.")
    if start_time and end_time and start_time > end_time:
        raise ValueError("'start_time' must be earlier than or equal to 'end_time'.")

    if input_path.is_file():
        if input_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip archive, got {input_path.name}")
        zip_files: List[Path] = [input_path]
    else:
        zip_files = _collect_zip_files(input_path)

    if not zip_files:
        raise FileNotFoundError(f"No .zip archives found under {input_path}")

    if start_time and end_time:
        wanted_codes = set(_get_timetable_periods(start_time, end_time))
        zip_files = [p for p in zip_files if p.stem in wanted_codes]

    if not zip_files:
        raise FileNotFoundError(
            "No CIF zips match the supplied date window "
            f"{start_time:%Y-%m-%d}-{end_time:%Y-%m-%d} in {input_path}"
        )

    log.info("Processing %d CIF zip file(s) from %s", len(zip_files), input_path)

    dataframes = (
        extract_CIF(p, cache_path=None) for p in zip_files
    )
    df = pd.concat(dataframes, ignore_index=True)
    log.info("Total hops extracted: %d", len(df))

    if cache_path:
        write_cache(Path(cache_path), df)

    return df

def get_timetable(cache_path: Union[str, Path] , input_path: Union[str, Path],start_time: dt.datetime = None, end_time: dt.datetime = None):
    """Return the timetable DataFrame from ``cache_path`` or build it."""
    if cache_path and Path(cache_path).exists():
        return get_cache(cache_path)
    return extract_timetable(
        start_time=start_time,
        end_time=end_time,
        input_path=input_path,
        cache_path=cache_path,
)