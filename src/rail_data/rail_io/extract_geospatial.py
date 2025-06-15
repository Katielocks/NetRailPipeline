from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Union

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
    """Generate geospatial buckets around a location code.

    Parameters
    ----------
    location_code
        Column in ``loc_df`` identifying the location code to use when linking
        to the ELR track data.
    seg_len_mi
        Length of track segments to extract, in miles.
    max_distance_m
        Maximum distance from the location to include, in metres.
    cache_path
        Path to store the resulting GeoDataFrame.
    loc_df
        DataFrame of location reference data.
    track_shp
        Loaded track shapefile as a :class:`geopandas.GeoDataFrame`.
    Returns
    -------
    pandas.DataFrame
        DataFrame describing each extracted geospatial bucket and location id mapping.
    """

    geospatial_buckets = link_loc_to_elr(loc_df,
                              track_shp,
                              loc_col=location_code,
                              seg_length=seg_len_mi,
                              max_distance_m=max_distance_m
                              )
    if cache_path:
        write_cache(cache_path,geospatial_buckets)
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
    """Return or create geospatial buckets describing track around locations.

    Parameters take precedence over values defined in :mod:`settings` when
    provided.  If a cache exists at ``cache_path`` it is returned; otherwise the
    location and track data are loaded and :func:`extract_geospatial` is used to
    generate the dataset.

    Parameters
    ----------
    location_code
        Field in the location DataFrame identifying the location code column.
    seg_len_mi
        Length of track segments in miles.
    max_distance_m
        Maximum distance from the location to include, in metres.
    cache_path
        Destination for the cached geospatial buckets.
    location_cache, location_input
        Paths used to obtain the location reference DataFrame.
    track_input
        Path to the track model (directory, ``.shp`` or ``.zip``).

    Returns
    -------
    pandas.DataFrame
        DataFrame produced by :func:`extract_geospatial`.
    """
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
    return extract_geospatial( location_code = location_code,
                        seg_len_mi = seg_len_mi,
                        max_distance_m = max_distance_m,
                        cache_path = cache_path,
                        loc_df = loc_df,
                        track_shp = track
                    )
