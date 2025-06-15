from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Callable, Union

import pandas as pd
import geopandas as gpd

from loc_client import extract_location_codes
from loc2elr import link_loc_to_elr
from track_client import get_track
from utils import write_cache, get_cache
from config import settings


def extract_geospatial( location_code: str,
                    seg_len_mi : int,
                    max_distance_m : float,
                    cache_path: Union[str, Path],
                    loc_df: pd.DataFrame,
                    track_shp: gpd.geodataframe
                    ) -> pd.DataFrame:
    

    geospatial_buckets = link_loc_to_elr(loc_df,
                              track_shp,
                              loc_col=location_code,
                              seg_length=seg_len_mi,
                              max_distance_m=max_distance_m
                              )
    if cache_path:
        write_cache(geospatial_buckets, cache_path)
    return geospatial_buckets

def get_geospatial(
                   location_code: str | None = None,
                   seg_len_mi : int | None = None,
                   max_distance_m : float | None = None,
                   *,
                   cache_path: Union[str, Path] | None = None,
                   location_cache: Union[str, Path] | None = None,
                   location_input: Union[str, Path] | None = None,
                   track_input: Union[str, Path] | None = None):
    
    if settings:
        if settings.geospatial:
            cache_path = cache_path or settings.geospatial.cache
            location_code = location_code or settings.geospatial.loc_id_field
            max_distance_m = max_distance_m or settings.geospatial.max_distance_m
            seg_len_mi = seg_len_mi or settings.geospatial.seg_length_mi
        if settings.ref.track_model:
            track_input = track_input or settings.ref.track_model.input
        if settings.ref.netrail_loc:
            location_cache = location_cache or settings.ref.netrail_loc.cache
            location_input = location_input or settings.ref.netrail_loc.input

    if cache_path and Path(cache_path).exists():
        return get_cache(cache_path)
    
    with get_track(track_input) as track_path:
        track = gpd.read_file(track_path)
        if track.crs is None:
            track.set_crs("EPSG:27700", inplace=True)
    loc_df = get_cache(location_cache,location_input,extract_location_codes)
    extract_geospatial( location_code = location_code,
                        seg_len_mi = seg_len_mi,
                        max_distance_m = max_distance_m,
                        cache_path = cache_path,
                        loc_df = loc_df,
                        track_shp = track
                    )
