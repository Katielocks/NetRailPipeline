import datetime as dt
from pathlib import Path
from typing import Tuple, Union, DefaultDict

import numpy as np
import pandas as pd
from pyproj import Transformer
import logging

from midas_client import download_locations
from .utils import read_cache,write_cache
from .config import settings

log = logging.getLogger(__name__)

_TRANSFORMER = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def _osgb_to_wgs84_vec(easting: np.ndarray, northing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised conversion from OSGB‑36 (EPSG:27700) to WGS‑84 (EPSG:4326)."""
    return _TRANSFORMER.transform(easting, northing)

def _get_centroids(geospatial:pd.DataFrame) -> pd.DataFrame:
    if geospatial.empty:
        raise ValueError("`geospatial` must contain at least one row.")

    required_cols = {"EASTING", "NORTHING"}
    missing = required_cols - set(geospatial.columns)
    if missing:
        raise KeyError(f"`geospatial` missing required columns: {sorted(missing)}")
    
    key = geospatial.columns[0]

    centroid = (
        geospatial.groupby(key, as_index=False)[["EASTING", "NORTHING"]]
        .mean()
    )

    lon, lat = _osgb_to_wgs84_vec(
        centroid["EASTING"].to_numpy(),
        centroid["NORTHING"].to_numpy(),
    )

    centroid = centroid.assign(Latitude=lat, Longitude=lon)
    return centroid


def extract_weather(
    geospatial: pd.DataFrame,
    years: range| list[str],
    tables: dict[str, list[str]] | None,
    *,
    cache_dir: str | Path | None,
    cache_format: str | None,
    version: str = None,
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
    if not isinstance(geospatial,pd.DataFrame):
        raise ValueError("geospatial must be a pandas dataframe")
    
    if settings:
        if settings.weather:
            cache_dir = cache_dir or settings.weather.cache_dir
            cache_format = cache_format or settings.weather.cache_format
            if settings.weather.midas:
                tables = tables or settings.weather.midas.tables
                version = version or settings.weather.midas.version

    if not cache_dir or not cache_format:
        raise ValueError("Must provide Valid Cache Directory and Cache Format")
    centroid = _get_centroids(geospatial)
    station_map = download_locations(
        centroid,
        years=years,
        tables=tables,
        out_dir=cache_dir,
        out_fmt=cache_format,
    )

    return station_map


def get_weather(
    geospatial: Union[pd.DataFrame,Union[Path,str]],
    start_date: dt.date | str,
    end_date: dt.date | str,
    tables: dict[str, list[str]] | None = None,
    *,
    version: str | None = None,
    cache_dir: str | Path | None = None,
    cache_format: str | None = None,
) -> pd.DataFrame:
    """Retrieve weather data, downloading any missing tables.

    The function looks for cached MIDAS tables under ``cache_dir``.  If some of
    the requested ``tables`` or ``years`` are absent, it delegates to
    :func:`extract_weather` to download them.  ``geospatial`` may be either the
    DataFrame returned by :func:`get_geospatial` or a path to its cache.

    Parameters
    ----------
    geospatial
        Geospatial DataFrame or path to its cache.
    start_date, end_date
        Inclusive date range for which weather data should be ensured.
    tables
        Mapping of MIDAS table names to lists of observation columns.
    version
        MIDAS dataset version.
    cache_dir, cache_format
        Directory and file format used for caching.

    Returns
    -------
    pandas.DataFrame
        Combined station mapping for all requested tables/years.
    """
    years = [str(y) for y in range(start_date.year, end_date.year + 1)]

    station_map_path = cache_dir / "station_map.json"
    if station_map_path.exists():
        existing_map = read_cache(station_map_path)
    else:
        existing_map = pd.DataFrame()

    missing_caches: dict[str, list[int]] = DefaultDict(list)
    if cache_dir:
        cache_dir = Path(cache_dir)
        for tbl in tables:
            for yr in years:
                if not (cache_dir / f"{tbl}_{yr}.{cache_format}").exists():
                    missing_caches[tbl].append(yr)

    if not missing_caches:
        log.info("All weather files available in cache, if looking to redownload, call extract_weather")
        return None
    if settings and settings.geospatial and not isinstance(geospatial,pd.DataFrame):
        geospatial = geospatial or settings.geospatial.cache
    if isinstance(geospatial,(Path,str)):   
        geospatial = read_cache(geospatial)
    elif geospatial is None:
        raise ValueError("You must import the geospatial dataframe or provide a path to the cache")


    station_maps: list[pd.DataFrame] = []
    if not existing_map.empty:
        station_maps.append(existing_map)

    for tbl, yrs in missing_caches.items():
        if yrs:
            station_maps.append(
                extract_weather(
                    geospatial,
                    yrs,
                    {tbl: tables[tbl]},
                    cache_dir=cache_dir,
                    cache_format=cache_format,
                    version=version
                )
            )
    combined = (
        pd.concat(station_maps, ignore_index=True)
        .drop_duplicates(subset=["loc_id", "year"], keep="last")
        .sort_values(["loc_id", "year"])
        .reset_index(drop=True)
    )
    write_cache(station_map_path,combined)
    return combined

