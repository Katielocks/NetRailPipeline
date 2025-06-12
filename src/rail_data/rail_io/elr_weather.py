from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Callable, Union

import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import Transformer

from loc_client import get_location_codes
from loc2elr import link_loc_to_elr
from midas_client import download_locations
from track_client import get_track
from utils import write_cache, read_cache
from config import settings

cfg = settings.elr_loc

_TRANSFORMER = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

def _osgb_to_wgs84(easting: float, northing: float) -> Dict[str, float]:
    """Convert OSGB coordinates (EPSG:27700) to WGS‑84 lon/lat."""
    lon, lat = _TRANSFORMER.transform(easting, northing)
    return {"Latitude": lat, "Longitude": lon}


def get_loc_elr(cache_path: Union[str, Path] = cfg.cache,
                location_code: str = cfg.location_code,
                max_distance_m : float = cfg.max_distance_m,
                seg_len_mi : int = cfg.seg_len_mi,
                track_model_input: Union[str, Path] = settings.ref.track_model
                ) -> pd.DataFrame:
    """Return Location to ELR lookup, reading/writing a local cache when available."""
    if cache_path and Path(cache_path).exists():
        return read_cache(cache_path)

    with get_track(track_model_input) as track_path:
        track = gpd.read_file(track_path)
        if track.crs is None:
            track.set_crs("EPSG:27700", inplace=True)

    loc_df = get_location_codes()
    loc_elr = link_loc_to_elr(loc_df,
                              track,
                              loc_col=location_code,
                              seg_length=seg_len_mi,
                              max_distance_m=max_distance_m
                              )

    if cache_path:
        write_cache(loc_elr, cache_path)

    return loc_elr


def get_weather(
    locations: pd.DataFrame,
    years: List[str],
    cfg: Any = settings.weather,
) -> None:
    """Download MIDAS weather data for *locations* over *years*.

    Parameters
    ----------
    locations : pd.DataFrame
        Must contain columns ``location_code``, ``Longitude``, and ``Latitude``.
    years : list[str]
        Calendar years as strings, e.g. ``["2023", "2024"]``.
    cfg : Any, optional
        A settings dataclass (defaults to ``settings.weather``).
    """
    midas_cfg = cfg.midas
    download_locations(
        locations=locations,
        years=years,
        tables=midas_cfg.tables,
        columns_per_table=midas_cfg.columns,
        out_dir=cfg.cache_dir,
    )


def get_loc_weather(
    start_date: dt.datetime,
    end_date: dt.datetime,
    loc_elr: Union[Path,pd.DataFrame] = settings.elr_loc.cache,
    cfg: Any = settings.weather,
) -> None:
    """High‑level helper that downloads weather for each ELR location code.

    """ 
    if isinstance(loc_elr,Path):
        loc_elr = get_loc_elr(loc_elr)
    
    if loc_elr.empty:
        raise ValueError("`loc_elr` must contain at least one row.")

    required_cols = {"location_code", "Easting", "Northing"}
    missing = required_cols - set(loc_elr.columns)
    if missing:
        raise KeyError(f"`loc_elr` missing required columns: {missing}")

    centroid = (
        loc_elr
        .groupby("location_code", as_index=False)[["Easting", "Northing"]]
        .mean()
    )

    records: List[Dict[str, Any]] = []
    for _, row in centroid.iterrows():
        coords = _osgb_to_wgs84(row["Easting"], row["Northing"])
        records.append({"location_code": row["location_code"], **coords})

    loc_coords = pd.DataFrame.from_records(records)

    if start_date > end_date:
        raise ValueError("`start_date` must be on or before `end_date`.")

    years = [str(year) for year in range(start_date.year, end_date.year + 1)]


    get_loc_weather(locations=loc_coords, years=years, cfg=cfg)