from __future__ import annotations

import gzip
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List,Union

import pandas as pd
import numpy as np


from utils import read_cache, write_cache
from config import settings
log = logging.getLogger(__name__)


_FIELDS = [
    "RECORD_TYPE",
    "ACTION_CODE",
    "TIPLOC",
    "LOC_NAME",
    "START_DATE",
    "END_DATE",
    "OS_EASTING",
    "OS_NORTHING",
    "TIMING_POINT_TYPE",
    "ZONE",
    "STANOX",
    "OFF_NETWORK",
    "LPB",
]


class BplanError(RuntimeError):
    """Raised whenever we can't fetch/unpack/parse the BPLAN doc."""


def _parse_loc_records(file_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    open_func = gzip.open if file_path.suffix == ".gz" else open
    mode = "rt" if file_path.suffix == ".gz" else "r"

    with open_func(file_path, mode=mode, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if not raw.startswith("LOC"):
                continue

            parts = raw.rstrip("\n").split("\t")
            row = {k: v for k, v in zip(_FIELDS, parts)}
            records.append(row)

    return records




    


def extract_location_codes(input_path: Union[str, Path] = None, cache_path: Union[str, Path] = None) -> pd.DataFrame:
    """Parse the Network Rail BPLAN archive into a DataFrame.

    Parameters
    ----------
    input_path
        Path to the BPLAN ``.zip`` archive.
    cache_path
        Optional path where the resulting DataFrame will be cached.

    Returns
    -------
    pandas.DataFrame
        Cleaned location reference records.
    """
    if settings and settings.ref.netrail_loc:
        input_path = settings.ref.netrail_loc.input
        cache_path = settings.ref.netrail_loc.cache

    try:
        zip_path = Path(input_path).expanduser().resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"Local BPLAN not found at {zip_path}")
    except Exception as exc:
        raise BplanError(f"Could not locate BPLAN: {exc}") from exc

    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            candidate = None
            for m in members:
                lower = m.lower()
                if lower.endswith(".txt.gz") or lower.endswith(".txt"):
                    candidate = m
                    break
            if candidate is None:
                raise BplanError(
                    f"No .txt or .txt.gz found inside {zip_path.name}"
                )

            with tempfile.TemporaryDirectory() as td:
                extracted = zf.extract(candidate, path=td)
                loc_records = _parse_loc_records(Path(extracted))
    except Exception as exc:
        raise BplanError(f"Failed to unpack/parse BPLAN: {exc}") from exc

    try:
        df = pd.DataFrame(loc_records)
        df["OS_EASTING"] = pd.to_numeric(df["OS_EASTING"], errors="coerce")
        df["OS_NORTHING"] = pd.to_numeric(df["OS_NORTHING"], errors="coerce")
        df["STANOX"] = df["STANOX"].replace(r"^\s*$", np.nan, regex=True)

        df = df[df["STANOX"].notna()]
        df = df[df["OS_EASTING"].notna() & df["OS_EASTING"].ne(0) & df["OS_EASTING"].ne(999_999)]
    except Exception as exc:
        raise BplanError(f"Error processing BPLAN DataFrame: {exc}") from exc


    if cache_path:
        write_cache(cache_path,df)
    return df

def get_location_codes(input_path: Union[str, Path] = None, cache_path: Union[str, Path] = None) -> pd.DataFrame:
    """Return cached location codes or extract them from ``input_path``."""
    if settings and settings.ref.netrail_loc:
        input_path = settings.ref.netrail_loc.input
        cache_path = settings.ref.netrail_loc.cache
        
    if cache_path and cache_path.exists():
        return read_cache(cache_path)
    else:
        return extract_location_codes(input_path, cache_path)

