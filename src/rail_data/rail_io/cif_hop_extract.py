# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import logging
import zipfile
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, TextIO, Union

import pandas as pd
from tqdm import tqdm

from config import settings
from utils import write_cache

__all__ = [
    "Hop",
    "iter_cif_lines",
    "build_tiploc_maps",
    "iter_hops",
    "write_hops",
    "get_timetable_df",
    "TimetableError",
    "UnsupportedCacheFormatError",
    "UnsupportedOutputFormatError",
    "ZipFileNotFoundError",
]

cfg = settings.timetable

log = logging.getLogger(__name__)
DEFAULT_ENCODING = "latin-1"


def slice_field(rec: str, start: int, length: int) -> str:
    return rec[start - 1 : start - 1 + length]


def _normalize_time(raw: str) -> Optional[str]:
    t = raw.strip()
    if not (1 <= len(t) <= 5 and t.isdigit()):
        return None
    t = t.zfill(4)
    hh, mm = int(t[:2]), int(t[2:])
    return f"{hh:02d}{mm:02d}" if 0 <= hh < 24 and 0 <= mm < 60 else None


def _seconds_since_midnight(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[2:])

@dataclass(frozen=True, slots=True)
class Hop:
    train_id: str
    train_service_id: str
    stanox_dep: str
    stanox_arr: str
    daysofweek: str
    dep_time: str
    start_date: str
    end_date: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)

def iter_cif_lines(zip_path: Path | str, encoding: str = DEFAULT_ENCODING) -> Iterator[str]:
    p = Path(zip_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with zipfile.ZipFile(p) as zf:
        for name in sorted(zf.namelist()):
            if not name.upper().endswith((".MCA", ".CFA",".CIF")):
                continue
            log.debug("reading %s", name)
            with zf.open(name) as fh:
                wrapper: TextIO = io.TextIOWrapper(fh, encoding=encoding, newline="")
                yield from wrapper


def build_tiploc_maps(lines: Iterable[str]) -> tuple[Dict[str, str], Dict[str, str]]:
    tip2crs: Dict[str, str] = {}
    tip2stanox: Dict[str, str] = {}
    for ln in lines:
        if ln[:2] not in {"TI", "TA"}:
            continue
        tip = slice_field(ln, 3, 7).rstrip()
        stanox = slice_field(ln, 45, 5).strip()
        crs = slice_field(ln, 54, 3).strip()
        if crs:
            tip2crs[tip] = crs
        if stanox and stanox != "00000":
            tip2stanox[tip] = stanox
    return tip2crs, tip2stanox


def iter_hops(
    lines: Iterable[str],
    tip2stanox: Dict[str, str],
    *,
    progress: bool = True,
) -> Iterator[Hop]:
    last_dep_time: Optional[str] = None
    last_stanox: Optional[str] = None
    train_id = train_service_code = ""
    days_run = "0000000"
    run_from = run_to = ""
    for ln in tqdm(lines, desc="CIF records", disable=not progress):
        rec = ln[:2]
        if rec == "BS":
            train_id = slice_field(ln, 4, 6)
            train_service_code = slice_field(ln, 42, 8)
            days_run = slice_field(ln, 22, 7)
            run_from = slice_field(ln, 10, 6)
            run_to = slice_field(ln, 16, 6)
            last_dep_time = last_stanox = None
            continue
        if rec not in ("LO", "LI", "LT"):
            continue
        tip = slice_field(ln, 3, 7).rstrip()
        stanox = tip2stanox.get(tip)
        if not stanox:
            continue
        arr_hhmm = dep_hhmm = None
        if rec == "LO":
            dep_hhmm = _normalize_time(slice_field(ln, 11, 4))
        elif rec == "LI":
            arr_hhmm = _normalize_time(slice_field(ln, 11, 4))
            dep_hhmm = _normalize_time(slice_field(ln, 16, 4))
        if last_stanox and last_dep_time and arr_hhmm:
            adj_days = days_run
            if _seconds_since_midnight(dep_hhmm) < _seconds_since_midnight(last_dep_time):
                adj_days = adj_days[-1] + adj_days[:-1]
            yield Hop(
                train_id=train_id,
                train_service_id=train_service_code,
                stanox_dep=last_stanox,
                stanox_arr=stanox,
                daysofweek=adj_days,
                dep_time=last_dep_time,
                start_date=run_from,
                end_date=run_to,
            )
        if dep_hhmm:
            last_dep_time = dep_hhmm
            last_stanox = stanox


def write_hops(
    hops: Iterable[Hop],
    *,
    out: Path | str | None = None,
    fmt: str = "csv",
    quoting: int = csv.QUOTE_ALL,
    dtype_str: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame([h.as_dict() for h in hops], dtype="string" if dtype_str else None)
    if out is None:
        return df
    else:
        out_path = Path(out)
        write_cache(out_path,df)
        return df
    




def extract_CIF(zip_path: Union[str, Path], cache_path: Union[str, Path] = None) -> pd.DataFrame:
    log.info("parsing CIF zip: %s",cache_path)
    lines_mem = list(iter_cif_lines(zip_path))
    _, tip2stanox = build_tiploc_maps(lines_mem)
    hops_iter = iter_hops(lines_mem, tip2stanox)
    df = write_hops(
        hops_iter,
        out= cache_path
    )
    log.info("%d hops extracted", len(df))
    return df
