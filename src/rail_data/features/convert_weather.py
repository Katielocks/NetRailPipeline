import pandas as pd
import logging
import re
import numpy as np
from pathlib import Path
from dateutil import parser
import datetime as dt
from typing import Dict, List, Callable, Union

log = logging.getLogger(__name__)

from ..io import read_cache
from .config import settings
from .utils import write_to_parquet,sep_datetime



AGG_FUNCS: Dict[str, Callable] = {
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "mean": np.mean,  
}
_DATE_COMPENENTS = ["year", "month", "day", "hour"]

def _explode_hourly(df: pd.DataFrame,
                    time_col: str = "meto_stmp_time",
                    id_col: str = "src_id") -> pd.DataFrame:
    """
    Create a full hourly series for every weather station.
    One row per (station, hour) carrying forward the latest
    min/max temperature seen.
    """

    df = df.copy()

    corrupt = [c for c in df.columns if c.startswith("Unnamed")]
    if corrupt:
        df = df.drop(columns=corrupt)
    df[time_col] = pd.to_datetime(df[time_col],
                                  errors="coerce",
                                  utc=True)
    df[time_col] = df[time_col].dt.round("h")
    df = (df.sort_values(time_col)
            .drop_duplicates(subset=[id_col, time_col], keep="last"))

    hourly = (
    df.groupby(id_col, group_keys=False)            
      .apply(
          lambda g: (
              g.set_index(time_col)              
               .resample("h")                      
               .ffill()                              
               .assign(**{id_col: g.name})         
          ),
          include_groups=False                     
      )
      .reset_index()                                
      [[time_col, id_col]                           
       + [c for c in df.columns
          if c not in (time_col, id_col)]]
    )
    hourly[time_col] = hourly[time_col].dt.tz_localize(None)
    hourly = hourly.reset_index(drop=True) 

    return hourly

def _load_table(year:str,
                table:str,
                input_dir: Union[Path,str] = None,
                input_fmt: str = None):
    if settings and settings.weather:
        input_dir = input_dir or settings.weather.cache_dir
        input_fmt = input_fmt or settings.weather.cache_format
    
    if input_dir and Path(input_dir).exists():
        input_path = f"{input_dir}/{table}_{year}.{input_fmt}"
        log.debug("Loading weather table %s", input_path)
        df = read_cache(input_path)
        df["meto_stmp_time"] = pd.to_datetime(df["meto_stmp_time"],
                                    errors="coerce")
        df = df.fillna(0)

        return df
       

def _get_years(
    input_dir: Union[Path, str] = None,
    input_fmt: str = None,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None
) -> set[str]:
    
    input_dir = Path(input_dir or settings.weather.cache_dir)
    input_fmt = input_fmt or settings.weather.cache_format

    pattern = f"*_*.{input_fmt}"
    
    years: set[str] = set()
    year_rx = re.compile(r'_(\d{4})$')

    for path in input_dir.glob(pattern):
        stem = path.stem          
        m = year_rx.search(stem)
        if m:
            years.add(m.group(1))
        if start_date is not None and end_date is not None:
            y0, y1 = start_date.year, end_date.year
            years = {y for y in years if y0 <= int(y) <= y1}

    return {str(y) for y in years}




def build_raw_weather_feature_frame(
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    parquet_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Explodes data from Station spefic Dataset to a location spefic Dataset
    """
    log.info(
        "Building raw weather features from %s to %s", start_date, end_date
    )

    station_map = read_cache(f"{settings.weather.cache_dir}/station_map.json")
    station_map["year"] = station_map["year"].astype(str)
    
    if isinstance(start_date, str):
        start_date = parser.parse(start_date)
    
    if isinstance(end_date, str):
        end_date = parser.parse(end_date)

    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date   = pd.to_datetime(end_date).tz_localize(None)

    years = _get_years(start_date=start_date, end_date=end_date)
    for yr in years:
        log.debug("Processing weather year %s", yr)
        location = pd.DataFrame()
        for table_name, col_map in settings.weather.features.tables.items():

            raw = _load_table(yr, table_name)
            if start_date is not None and end_date is not None:
                buffer_time  = dt.timedelta(days=1)
                raw = raw[raw["meto_stmp_time"].between(start_date-buffer_time, end_date+buffer_time)]

            raw = _explode_hourly(raw)
            
            if start_date is not None and end_date is not None:
                raw = raw[raw["meto_stmp_time"].between(start_date, end_date)]
            
            dt_parts = sep_datetime(raw["meto_stmp_time"], _DATE_COMPENENTS)
            raw = pd.concat([raw,dt_parts],axis=1)
        
            src_col = f"src_id_{table_name}"
            raw = raw.rename(columns={"src_id": src_col})
            

            cols = _DATE_COMPENENTS + [src_col] + list(raw.columns[2:-4])
            df_tab = raw[cols].copy()
            df_tab["year"] = df_tab["year"].astype(str)
            
            df_loc = df_tab.merge(
                station_map[["loc_id", src_col, "year"]],
                on=[src_col, "year"],
                how="inner"
            ).drop(columns=[src_col])

            join_keys = ["loc_id"] + _DATE_COMPENENTS
            if location.empty:
                location = df_loc
            else:
                location = location.merge(
                    df_loc,
                    on=join_keys,
                    how="outer"
                )
        location = location.rename(columns={"loc_id": "ELR_MIL"})
        out_dir = parquet_dir or settings.weather.parquet_dir
        write_to_parquet(location, out_dir)
        log.info("Wrote base weather features for %s", yr)
            
