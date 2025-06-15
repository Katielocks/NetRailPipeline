from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd

from . import (
    extract_corpus,
    extract_delay_dataset,
    extract_geospatial,
    extract_location_codes,
    extract_timetable,
    extract_weather,
    fetch_public_holidays,
    get_corpus,
    get_delay_dataset,
    get_geospatial,
    get_location_codes,
    get_timetable,
    get_weather,
)
from .config import settings
from .delay_processer import _build_business_period_map, _YEAR_START
from .utils import write_cache
from .track_client import get_track


def _as_datetime(val: dt.date | dt.datetime | str) -> dt.datetime:
    if isinstance(val, dt.datetime):
        return val
    if isinstance(val, dt.date):
        return dt.datetime.combine(val, dt.time.min)
    if isinstance(val, str):
        return dt.datetime.fromisoformat(val)
    raise TypeError(f"Unsupported date type: {type(val)!r}")


def extract_datasets(start_date: dt.date | dt.datetime | str, end_date: dt.date | dt.datetime | str) -> None:
    """Extract all datasets within the given date range.

    Parameters
    ----------
    start_date : datetime.date | datetime.datetime | str
        Inclusive start of the range. If a string is provided it must be ISO
        formatted.
    end_date : datetime.date | datetime.datetime | str
        Inclusive end of the range. If a string is provided it must be ISO
        formatted.

    Existing caches will be overwritten.
    """
    start_dt = _as_datetime(start_date)
    end_dt = _as_datetime(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    loc_df = extract_location_codes(settings.ref.netrail_loc.input, settings.ref.netrail_loc.cache)

    with get_track(settings.ref.track_model.input) as shp:
        track = gpd.read_file(shp)
        if track.crs is None:
            track.set_crs("EPSG:27700", inplace=True)

    geo_df = extract_geospatial(
        location_code=settings.geospatial.loc_id_field,
        seg_len_mi=settings.geospatial.seg_length_mi,
        max_distance_m=settings.geospatial.max_distance_m,
        cache_path=settings.geospatial.cache,
        loc_df=loc_df,
        track_shp=track,
    )

    years = [str(y) for y in range(start_dt.year, end_dt.year + 1)]
    extract_weather(
        geo_df,
        years,
        settings.weather.midas.tables,
        cache_dir=settings.weather.cache_dir,
        cache_format=settings.weather.cache_format,
        version=settings.weather.midas.version,
    )

    extract_timetable(
        start_time=start_dt,
        end_time=end_dt,
        input_path=settings.timetable.input,
        cache_path=settings.timetable.cache,
    )

    periods = _build_business_period_map(start_dt, end_dt, _YEAR_START)
    extract_delay_dataset(
        business_period=periods,
        overwrite=True,
        src_dir=Path(settings.delay.input),
        out_dir=Path(settings.delay.cache),
        out_format=settings.delay.cache_format,
    )

    fetch_public_holidays()

    corpus_df = extract_corpus(settings.ref.corpus.input)
    if settings.ref.corpus.cache:
        write_cache(settings.ref.corpus.cache, corpus_df)


def get_datasets(start_date: dt.date | dt.datetime | str, end_date: dt.date | dt.datetime | str) -> None:
    """Ensure all datasets between ``start_date`` and ``end_date`` are available.

    Parameters
    ----------
    start_date : datetime.date | datetime.datetime | str
        Inclusive start of the range. Strings should be ISO formatted.
    end_date : datetime.date | datetime.datetime | str
        Inclusive end of the range. Strings should be ISO formatted.

    Missing caches will be populated and existing ones are preserved.
    """
    start_dt = _as_datetime(start_date)
    end_dt = _as_datetime(end_date)
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    get_location_codes(settings.ref.netrail_loc.input, settings.ref.netrail_loc.cache)

    geo_df = get_geospatial()

    get_weather(
        geospatial=geo_df,
        start_date=start_dt,
        end_date=end_dt,
        tables=settings.weather.midas.tables,
        version=settings.weather.midas.version,
        cache_dir=settings.weather.cache_dir,
        cache_format=settings.weather.cache_format,
    )

    get_timetable(
        cache_path=settings.timetable.cache,
        input_path=settings.timetable.input,
        start_time=start_dt,
        end_time=end_dt,
    )

    get_delay_dataset(
        start_date=start_dt,
        end_date=end_dt,
        src_dir=Path(settings.delay.input),
        out_dir=Path(settings.delay.cache),
        out_format=settings.delay.cache_format,
    )

    if settings.ref.bank_holiday.cache and not Path(settings.ref.bank_holiday.cache).exists():
        fetch_public_holidays()

    if settings.ref.corpus.cache and not Path(settings.ref.corpus.cache).exists():
        df = get_corpus(settings.ref.corpus.cache, settings.ref.corpus.input)
        if df is not None and not Path(settings.ref.corpus.cache).exists():
            write_cache(settings.ref.corpus.cache, df)

