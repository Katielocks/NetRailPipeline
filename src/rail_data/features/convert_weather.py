import pandas as pd
import logging
import re
import numpy as np
from pathlib import Path
import datetime as dt
from typing import Dict, List,Callable,Union
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
def _explode_hourly(df: pd.DataFrame, time_col: str = "meto_stmp_time") -> pd.DataFrame:
    df = df[df.notna()]
    df = df.set_index(time_col).resample("h").asfreq()   
    return df.ffill()


def _load_table(year:str,
                table:str,
                input_dir: Union[Path,str] = None,
                input_fmt: str = None):
    if settings and settings.weather:
        input_dir = input_dir or settings.weather.cache_dir
        input_fmt = input_fmt or settings.weather.cache_format
    
    
    if input_dir and Path(input_dir).exists():
       input_path = f"{input_dir}/{table}_{year}.{input_fmt}"
       return read_cache(input_path)
       

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
            years = {y for y in years if y0 <= y <= y1}

    return {str(y) for y in years}




def build_raw_weather_feature_frame(start_date: dt.datetime = None,end_date: dt.datetime = None) -> pd.DataFrame:
    """
    Explodes data from Station spefic Dataset to a location spefic Dataset
    """

    station_map = read_cache(f"{settings.weather.cache_dir}/station_map.json")
    station_map["year"] = station_map["year"].astype(str)
    years = _get_years(start_date=start_date,end_date=end_date)
    for yr in years:
        location = pd.DataFrame()
        for table_name, col_map in settings.weather.features.tables.items():

            raw = _load_table(yr, table_name)
            raw = _explode_hourly(raw)
            
            if start_date is not None and end_date is not None:
                raw = raw[raw["meto_stmp_time"].between(start_date, end_date)]
            
            dt_parts = sep_datetime(raw["meto_stmp_time"], _DATE_COMPENENTS)
            raw = pd.concat([raw,dt_parts],axis=1)
        
            src_col = f"src_id_{table_name}"
            raw = raw.rename(columns={"src_id": src_col})
            
            feature_cols = list(col_map.keys())
            cols = _DATE_COMPENENTS + [src_col] + feature_cols
            df_tab = raw[cols].copy()
            
            df_loc = df_tab.merge(
                station_map[["loc_id", src_col]],
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
        write_to_parquet(location,settings.weather.parquet_dir)
            
