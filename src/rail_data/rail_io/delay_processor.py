#!/usr/bin/env python
from __future__ import annotations
import logging
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Final, Mapping
import pandas as pd
from .utils import write_cache

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
_DROP_COLS: Final[list[str]] = [
    "INCIDENT_RESPONSIBLE_TRAIN",
    "INCIDENT_RESP_TRAIN",
    "REACT_TRAIN",
]

_PERIOD_ZIP_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i).*?"
    r"(?P<season>\d{4}(?:[-_ ]?\d{2})?|\d{2}[-_ ]\d{2})"  # season component
    r"[-_ ]*P(?P<period>\d{2})"  # business period
    r"\.zip$",
)


log = logging.getLogger(__name__)

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
      .pipe(lambda d: d.drop(
          columns=[c for c in _DROP_COLS if c in d.columns],
      ))
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
            write_cache(dest,df)
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


def process_zipfile(
    zip_path: Path,
    output_root: Path,
    output_format: str,
    *,
    overwrite: bool,
    business_periods: Mapping[str, set[str]] | None = None,
    import_all: bool = False,
) -> int:
    if _handle_period_zip(zip_path, output_root,output_format, business_periods=business_periods, import_all=import_all):
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
