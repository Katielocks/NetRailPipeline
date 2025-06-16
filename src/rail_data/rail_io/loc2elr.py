from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable,Union

import geopandas as gpd
import pandas as pd

from .bplan_client import get_bplan
from .track_client import get_elr
from .config import settings as cfg

__all__ = [
    "link_bplan_to_elr",
    "loc2elr"
]

log = logging.getLogger(__name__)


def link_loc_to_elr(
    loc_df: Union[pd.DataFrame,gpd.GeoDataFrame],
    track_gdf: gpd.GeoDataFrame,
    *,
    loc_col: str = cfg.model.loc_id_field,
    easting_col: str = "EASTING",
    northing_col: str = "NORTHING",
    elr_col: str = cfg.model.elr_id_field,
    start_col: str = cfg.model.start_id_field,
    max_distance_m: float = cfg.model.max_distance_m,
    seg_length: int = cfg.model.seg_length_mi
) -> pd.DataFrame:
    """
    Assigns each STANOX location from BPLAN to the nearest ELR track segment and computes bucket IDs.

    Parameters
    ----------
    loc_df : geopandas.GeoDataFrame,pd.Dataframe
        pd.Dataframe or GeoDataFrame containing columns for location ID, easting and northing.
    track_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing ELR track segments with ELR IDs and start mile markers.
    loc_col : str, optional
        Column name in bplan_gdf for the location identifier (default from config).
    easting_col : str, optional
        Column name for easting coordinates (default 'EASTING').
    northing_col : str, optional
        Column name for northing coordinates (default 'NORTHING').
    elr_col : str, optional
        Column name in track_gdf for the ELR identifier (default from config).
    start_col : str, optional
        Column name in track_gdf for the segment start mile marker (default from config).
    max_distance_m : float, optional
        Maximum search distance in meters for nearest neighbor join (default from config).
    seg_length : int, optional
        Segment length in miles for bucketing intervals (default from config).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing original bplan data plus nearest ELR segment info and
        'ELR_MIL' column combining ELR ID and bucketed start mile.
    """
        
    loc = loc_df.copy()
    track = track_gdf.copy()

    osgb = "EPSG:27700"
    if not isinstance(loc, gpd.GeoDataFrame):
        loc = gpd.GeoDataFrame(
            loc,
            geometry=gpd.points_from_xy(loc[easting_col], loc[northing_col]),
            crs=osgb,
        )


    if loc.crs != osgb:
        raise ValueError("loc_gdf CRS must be EPSG:27700")
    if track.crs != osgb:
        raise ValueError("track_gdf CRS must be EPSG:27700")

    missing_loc = {loc_col, easting_col, northing_col} - set(loc.columns)
    if missing_loc:
        raise KeyError(f"loc missing columns: {sorted(missing_loc)}")

    missing_track = {elr_col, start_col} - set(track.columns)
    if missing_track:
        raise KeyError(f"Track model missing columns: {sorted(missing_track)}")



    track = track[[elr_col, start_col, "geometry"]]

    joined = gpd.sjoin_nearest(
        loc,
        track,
        how="left",
        max_distance=max_distance_m,
        distance_col="DIST_M",
    )

    out_df = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    if seg_length:
        out_df[start_col] = (
        out_df[start_col]
        .fillna(0)
        .astype(int)
        .floordiv(seg_length)  
        .mul(seg_length)      
        )
    out_df = out_df[out_df[elr_col].notna()]
    out_df["ELR_MIL"] = (
        out_df[elr_col] + "_" + out_df[start_col].astype("str")
    )
    out_df[[loc_col,easting_col,northing_col,"ELR_MIL"]]
    return out_df


_OUTPUT_WRITERS: dict[str, Callable[[pd.DataFrame, Path], None]] = {
    "csv": lambda df, p: df.to_csv(p, index=False),
    "parquet": lambda df, p: df.to_parquet(p, index=False),
    "json": lambda df, p: df.to_json(p, orient="records"),
}


def loc2elr(
    *,
    bplan_source: str | Path | None = None,
    track_source: str | Path | None = None,
    output_path: str | Path | None = None,
    loc_col: str = cfg.model.loc_id_field,
    easting_col: str = "EASTING",
    northing_col: str = "NORTHING",
    elr_col: str = cfg.model.elr_id_field,
    start_col: str = cfg.model.start_id_field,
    max_distance_m: float = cfg.model.max_distance_m,
    seg_length_mi: int = cfg.model.seg_length_mi
) -> pd.DataFrame:
    """
    Orchestrates loading of loc and ELR data, computes ELR mile buckets, and writes output.

    Parameters
    ----------
    loc_source : str or Path, optional
        Path or identifier for the Bplan.zip. If None, uses config default.
    track_source : str or Path, optional
        Path or identifier for the NWR Track Model location or sho. If None, uses config default.
    output_path : str or Path, optional
        Path to write the resulting DataFrame. Supported extensions: .csv, .parquet, .json.
        If None, no file is written.
    loc_col : str, optional
        Column name for location ID in BPLAN data (default from config).
    easting_col : str, optional
        Column name for easting coordinates (default 'EASTING').
    northing_col : str, optional
        Column name for northing coordinates (default 'NORTHING').
    elr_col : str, optional
        Column name for ELR ID in track data (default from config).
    start_col : str, optional
        Column name for segment start mile in track data (default from config).
    max_distance_m : float, optional
        Maximum search distance in meters for nearest join (default from config).
    seg_length_mi : int, optional
        Segment length in miles for bucketing intervals (default from config).

    Returns
    -------
    pandas.DataFrame
        DataFrame with STANOX locations mapped to ELR mile buckets, matching link_bplan_to_elr output.
    """

    bplan_source = bplan_source or cfg.io.bplan
    track_source = track_source or cfg.io.track_model
                                                                   
    loc_df = get_bplan(bplan_source)

    with get_elr(track_source) as track_path:
        track_gdf = gpd.read_file(track_path)
        if track_gdf.crs is None:
            track_gdf.set_crs("EPSG:27700", inplace=True)

    out_df = link_loc_to_elr(
        loc_df= loc_df,
        track_gdf=track_gdf,
        loc_col=loc_col,
        easting_col=easting_col,
        northing_col=northing_col,
        elr_col=elr_col,
        start_col=start_col,
        max_distance_m=max_distance_m,
        seg_length=seg_length_mi
    )
                                                        
    if output_path is not None:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = out_path.suffix.lstrip(".").lower()
        try:
            _OUTPUT_WRITERS[fmt](out_df, out_path)
        except KeyError:
            raise ValueError(
                f"Unsupported output format ‘{fmt}’. Choose one of {list(_OUTPUT_WRITERS)}"
            ) from None
        log.info("Wrote %s records to %s", len(out_df), out_path)

    return out_df
