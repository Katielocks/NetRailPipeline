#!/usr/bin/env python
from __future__ import annotations
import math
import datetime as dt
import logging
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Final, Iterable, Mapping, MutableMapping, Sequence,Union,Dict,Set,List
import pandas as pd
from config import settings
from utils import write_cache


_YEAR_START: Final[tuple[int, int]] = (4, 1)  # (month, day) – UK rail FY begins 1 April
_PART_DURATION: Final[dt.timedelta] = dt.timedelta(days=28)

_COL_NAMES: Final[list[str]] = [
    "FINANCIAL_YEAR_PERIOD",
    "ORIGIN_DEPARTURE_DATE",
    "TRUST_TRAIN_ID",
    "PLANNED_ORIGIN_LOCATION_CODE",
    "PLANNED_ORIGIN_WTT_DATETIME",
    "PLANNED_ORIGIN_GBTT_DATETIME",
    "PLANNED_DEST_LOCATION_CODE",
    "PLANNED_DEST_WTT_DATETIME",
    "PLANNED_DEST_GBTT_DATETIME",
    "TRAIN_SERVICE_CODE",
    "SERVICE_GROUP_CODE",
    "TOC_CODE",
    "ENGLISH_DAY_TYPE",
    "APPLICABLE_TIMETABLE_FLAG",
    "TRAIN_SCHEDULE_TYPE",
    "TRACTION_TYPE",
    "TRAILING_LOAD",
    "TIMING_LOAD",
    "UNIT_CLASS",
    "INCIDENT_NUMBER",
    "INCIDENT_CREATE_DATE",
    "INCIDENT_START_DATETIME",
    "INCIDENT_END_DATETIME",
    "SECTION_CODE",
    "NR_LOCATION_MANAGER",
    "RESPONSIBLE_MANAGER",
    "INCIDENT_REASON",
    "ATTRIBUTION_STATUS",
    "INCIDENT_EQUIPMENT",
    "INCIDENT_DESCRIPTION",
    "REACT_REASON",
    "INCIDENT_RESP_TRAIN",
    "EVENT_TYPE",
    "START_STANOX",
    "END_STANOX",
    "EVENT_DATETIME",
    "PFPI_MINUTES",
]

_IMPORT_COLS: Final[list[str]] = [
    "INCIDENT_START_DATETIME",
    "INCIDENT_NUMBER",
    "SECTION_CODE",
    "INCIDENT_REASON",
    "PLANNED_ORIGIN_WTT_DATETIME",
    "EVENT_DATETIME",
    "EVENT_TYPE",
    "TRAIN_SERVICE_CODE",
    "REACT_REASON",
    "START_STANOX",
    "END_STANOX",
    "PFPI_MINUTES",
]

_DATE_COLS: Final[list[str]] = [
    "INCIDENT_START_DATETIME",
    "PLANNED_ORIGIN_WTT_DATETIME",
    "EVENT_DATETIME",
]

_PERIOD_ZIP_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i).*?"
    r"(?P<season>\d{4}(?:[-_ ]?\d{2})?|\d{2}[-_ ]\d{2})"  # season component
    r"[-_ ]*P(?P<period>\d{2})"  # business period
    r"\.zip$",
)


log = logging.getLogger(__name__)
cfg  = settings.delay

def _swap_columns(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    cols = list(df.columns)
    if col1 in cols and col2 in cols:
        i, j = cols.index(col1), cols.index(col2)
        cols[i], cols[j] = cols[j], cols[i]
    return df[cols]


def _infer_datetime_format(sample: pd.Series) -> str: 
    sample = sample.dropna().astype(str)
    if sample.empty:
        raise ValueError("No non‑null values to infer format from.")

    probe = sample.iloc[0].strip()

    m_sep = re.search(r"\d([-/])", probe)
    sep = m_sep.group(1) if m_sep else "/"

    parts = probe.split(sep)
    if len(parts) < 3:
        raise ValueError(f"Unexpected date layout in '{probe}'.")

    month_token = parts[1]
    is_alpha = bool(re.search(r"[A-Za-z]", month_token))

    fmt = f"%d{sep}%b{sep}%Y %H:%M" if is_alpha else f"%d{sep}%m{sep}%Y %H:%M"
    return fmt


def _process_delay_dataframe(handle) -> pd.DataFrame: 
    df = (
        pd.read_csv(handle, low_memory=False)
        .rename(columns=str.upper)
        .iloc[:, :37] 
    )
    df = _swap_columns(df, "PLANNED_ORIG_GBTT_DATETIME_AFF", "PLANNED_ORIG_WTT_DATETIME_AFF")
    df.columns = _COL_NAMES
    df = df[_IMPORT_COLS]

    date_format = _infer_datetime_format(df[_DATE_COLS[1]])
    for col in _DATE_COLS:
        df = df[df[col].notna()]
        df[col] = pd.to_datetime(df[col], format=date_format)

    return df

def _extract_first_csv(zip_path: Path, dest_dir: Path, fmt: str, period: str, *, name: str = "delay") -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            dest = dest_dir / period / f"{name}.{fmt}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src:
                df = _process_delay_dataframe(src)
            write_cache(dest_dir,df)
            log.info(" %s to %s", zip_path.name, dest.relative_to(dest_dir.parent))
            return
    log.warning(" No CSV found inside %s – skipped", zip_path.name)


def _handle_period_zip(
    zip_path: Path,
    output_root: Path,
    output_format: str,
    *,
    business_periods: Mapping[str, set[str]] | None = None,
    import_all: bool = False,
) -> bool:
    m = _PERIOD_ZIP_RE.match(zip_path.name)
    if not m:
        return False

    season_raw = m.group("season")
    period = f"P{m.group('period')}"

    # Normalise season key to YYYYMM
    season_key = season_raw.replace("-", "").replace("_", "").replace(" ", "")
    if len(season_key) < 6:
        season_key = f"20{season_key}"

    if not import_all and business_periods:
        if season_key not in business_periods or period not in business_periods[season_key]:
            return False

    season_dir = output_root / season_key

    try:
        _extract_first_csv(zip_path, season_dir, output_format,period)
    except zipfile.BadZipFile:
        log.error("✗ %s is corrupted – skipping", zip_path.name)
    return True


def _process_zipfile(
    zip_path: Path,
    output_root: Path,
    output_format: str,
    *,
    overwrite: bool,
    business_periods: Mapping[str, set[str]] | None = None,
    import_all: bool = False,
) -> int:
    if _handle_period_zip(zip_path, output_root, business_periods=business_periods, import_all=import_all):
        return 1

    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            log.warning("✗ %s is invalid – skipping outer ZIP", zip_path.name)
            return 0

        for inner_zip in Path(tmp).rglob("*.zip"):
            if _handle_period_zip(inner_zip, output_root,output_format, business_periods=business_periods, import_all=import_all):
                written += 1
    return written

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
    start_date: dt.datetime | None = None,
    end_date: dt.datetime | None = None,
    import_all: bool = False,
    overwrite: bool = False,
    src_dir: Path = cfg.input,
    out_dir: Path = cfg.cache,
    out_format: Path = cfg.cache_format,

) -> None:

    if not src_dir.is_dir():
        raise SystemExit(f"Input directory {src_dir} does not exist")

    out_dir.mkdir(parents=True, exist_ok=True)

    if start_date is None or end_date is None:
        import_all = True

    business_period_parts: MutableMapping[str, set[str]] = _build_business_period_map(
        start_date,
        end_date,
        _YEAR_START,
    )

    if not overwrite:
        business_period_parts = _prune_business_period_map(business_period_parts, out_dir, "csv", {"delay"})

    if not business_period_parts:
        log.info("Nothing to do – all requested datasets already cached")
        return

    raw_delay_check = _check_folder(src_dir, "zip", business_period_parts.keys())
    if raw_delay_check is not True:
        missing_years = ", ".join(str(x) for x in raw_delay_check)
        log.warning(" Missing raw datasets for: %s", missing_years)
        log.info("You can download ZIPs from https://raildata.org.uk/")
        for yr in raw_delay_check:
            business_period_parts.pop(yr, None)
        if not business_period_parts:
            log.info("No remaining datasets – exiting")
            return

    total = 0
    for year in sorted(business_period_parts):
        zip_path = src_dir / f"{year}.zip"
        log.info("Processing %s ...", zip_path.name)
        total += _process_zipfile(
            zip_path,
            out_dir,
            out_format,
            overwrite=overwrite,
            business_periods=business_period_parts,
            import_all=import_all,
        )

    


