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

cfg = settings.geospatial

netr_loc_cfg = settings.ref.netrail_loc
trk_mdl_cfg = settings.ref.track_model

def extract_spatio(cache_path: Union[str, Path] = cfg.cache,
                seg_len_mi : int = cfg.seg_len_mi,
                max_distance_m : float = cfg.max_distance_m,
                *,
                location_code: str = cfg.location_code,
                track_model_input: Union[str, Path] = trk_mdl_cfg.cache
                ) -> pd.DataFrame:
    
    with get_track(track_model_input) as track_path:
        track = gpd.read_file(track_path)
        if track.crs is None:
            track.set_crs("EPSG:27700", inplace=True)

    loc_df = get_cache(netr_loc_cfg.cache,netr_loc_cfg.input,extract_location_codes)

    geospatial_buckets = link_loc_to_elr(loc_df,
                              track,
                              loc_col=location_code,
                              seg_length=seg_len_mi,
                              max_distance_m=max_distance_m
                              )
    if cache_path:
        write_cache(geospatial_buckets, cache_path)
    return geospatial_buckets
