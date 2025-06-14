import datetime as dt
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

from midas_client import download_locations
from utils import read_cache
from config import settings

midas_cfg = settings.weather.midas
_geospatial_cfg_default = settings.geospatial



_TRANSFORMER = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def _osgb_to_wgs84_vec(easting: np.ndarray, northing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised conversion from OSGB‑36 (EPSG:27700) to WGS‑84 (EPSG:4326)."""
    return _TRANSFORMER.transform(easting, northing)


def extract_weather(
    geospatial: pd.DataFrame,
    start_date: dt.datetime | str,
    end_date: dt.datetime | str,
    tables: dict[str, list[str]] | None = midas_cfg.tables,
    *,
    cache_dir: str | Path | None = settings.weather.cache_dir,
    cache_format: str | None = settings.weather.cache_format,
) -> pd.DataFrame:
    

    """Download MIDAS weather data for the provided *geospatial* buckets.

    Parameters
    ----------
    geospatial
        DataFrame containing *at least* ``location_code``, ``Easting`` and ``Northing``.
    start_date, end_date
        Inclusive date range. Accepts ISO‑8601 strings or ``datetime``/``date`` objects.
    tables
        Dict of MIDAS obs table names; with observation columns eg (max_temp) as items.
    cache_dir, cache_format
        Override the default path/format in :data:`settings.weather`.

    Returns
    -------
    pandas.DataFrame
        Centroids (one per ``location_code``) with added ``Latitude``/``Longitude``.
    """

    if geospatial.empty:
        raise ValueError("`geospatial` must contain at least one row.")

    required_cols = {"location_code", "Easting", "Northing"}
    missing = required_cols - set(geospatial.columns)
    if missing:
        raise KeyError(f"`geospatial` missing required columns: {sorted(missing)}")


    if isinstance(start_date, str):
        start_date = dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = dt.date.fromisoformat(end_date)

    if start_date > end_date:
        raise ValueError("`start_date` must be on or before `end_date`.")

    years = [str(y) for y in range(start_date.year, end_date.year + 1)]
    centroid = (
        geospatial.groupby("location_code", as_index=False)[["Easting", "Northing"]]
        .mean()
    )

    lon, lat = _osgb_to_wgs84_vec(
        centroid["Easting"].to_numpy(),
        centroid["Northing"].to_numpy(),
    )

    centroid = centroid.assign(Latitude=lat, Longitude=lon)

    download_locations(
        centroid[["location_code", "Latitude", "Longitude"]],
        years=years,
        tables=tables,
        out_dir=cache_dir,
        out_fmt=cache_format,
    )

    return centroid


def get_weather(
    start_date: dt.date | str,
    end_date: dt.date | str,
    tables: Sequence[str] | Mapping[str, str] | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_format: str | None = None,
    geospatial_cfg: Any = None,
) -> pd.DataFrame:
    """High‑level convenience wrapper around :func:`extract_weather`.

    It loads the geospatial bucket cache referenced by *geospatial_cfg* (defaults to
    :data:`settings.geospatial`) and returns the DataFrame produced by
    :func:`extract_weather`.
    """

    cfg = geospatial_cfg or _geospatial_cfg_default

    if cfg.cache is None:
        raise ValueError("`geospatial_cfg.cache` is not set.")
    if not cfg.cache.exists():
        raise FileNotFoundError(f"Geo cache not found at {cfg.cache}")

    geospatial = read_cache(cfg.cache)

    return extract_weather(
        geospatial,
        start_date,
        end_date,
        tables=tables,
        cache_dir=cache_dir,
        cache_format=cache_format,
    )


