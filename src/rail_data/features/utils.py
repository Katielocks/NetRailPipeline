from __future__ import annotations

from pathlib import Path
from datetime import timedelta
from typing import Final, Iterable, Union, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import logging


from ..io import settings, get_geospatial

logger = logging.getLogger(__name__)

_PARTITION_COLS: Final[list[str]] = [
    "ELR_MIL", "year", "month", "day",
]

def sep_datetime(
    datetime_column: Union[pd.Series, pd.DatetimeIndex],
    components: List[str] = ["year", "month", "day", "hour"]
) -> pd.DataFrame:
    """
    Split a datetime series into separate components.

    Args:
        datetime_column: A pandas Series or DatetimeIndex of dtype datetime64.
    Returns:
        DataFrame with each requested component as a column.
        
    """
    if isinstance(datetime_column, pd.DatetimeIndex):
        series = pd.Series(datetime_column, name='datetime')
    elif isinstance(datetime_column, pd.Series):
        series = datetime_column.copy()
    else:
        logger.error("Provided datetime_column is not a pandas Series or DatetimeIndex.")
        raise ValueError("datetime_column must be a pandas Series or DatetimeIndex.")

    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            series = pd.to_datetime(series)
        except Exception as e:
            logger.exception("Failed to convert series to datetime: %s", e)
            raise ValueError("Could not convert series to datetime dtype.")
    supported = {
        'year': 'year',
        'month': 'month',
        'day': 'day',
        'hour': 'hour',
        'minute': 'minute',
        'second': 'second',
        'weekday': 'weekday' 
    }

    if components is None:
        to_extract = list(supported.keys())
        logger.info("No components list provided; extracting all: %s", to_extract)
    else:
        to_extract = components
        invalid = [c for c in to_extract if c not in supported]
        if invalid:
            logger.error("Unsupported datetime components requested: %s", invalid)
            raise ValueError(f"Unsupported components: {invalid}. "
                             f"Supported: {list(supported.keys())}")

    result = pd.DataFrame(index=series.index)
    for comp in to_extract:
        try:
            result[comp] = getattr(series.dt, supported[comp])
        except Exception as e:
            logger.exception("Error extracting component '%s': %s", comp, e)
            raise RuntimeError(f"Failed to extract component '{comp}'.")

    return result

def location_to_ELR_MIL(location_column:pd.Series, geo_df: pd.DataFrame = None) -> pd.Series:
    if not geo_df:
        if settings and settings.geospatial:
            geo_df = settings.geospatial.cache
        geo_df = get_geospatial(geo_df)
    mapping =  geo_df.set_index("STANOX")["ELR_MIL"].to_dict()
    return location_column.map(mapping)


def write_to_parquet(
    df: pd.DataFrame,
    out_root: str | Path,
    partition_cols: Iterable[str] = None,
    parquet_compression: str | None = "snappy",
    max_parts: int = 5000
) -> ds.Dataset:
    
    partition_cols = partition_cols or _PARTITION_COLS

    out_path = Path(out_root).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    try:
        pq.write_to_dataset(
            table,
            root_path=str(out_path),
            partition_cols=list(partition_cols),
            existing_data_behavior="overwrite_or_ignore",  
            compression=parquet_compression,
             max_partitions=max_parts,
        )
    except TypeError: 
        pq.write_to_dataset(
            table,
            root_path=str(out_path),
            partition_cols=list(partition_cols),
            compression=parquet_compression,
        )
