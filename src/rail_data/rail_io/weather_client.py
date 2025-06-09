from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
from datetime import datetime
import logging

from config import settings
from bplan_client import fetch_and_parse_bplan
from midas_client import download_locations
from utils import read_cache

__all__ = ["WeatherClient", "WeatherClientError"]

class WeatherClientError(RuntimeError):
    """Raised when the BPlan file cannot be found or weather data cannot be fetched."""


cfg = settings.weather
log = logging.getLogger(__name__)

class WeatherClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        bplan_path: str | Path | None = None,
    ) -> None:

        self._bplan_cache = Path(bplan_path or cfg.ref.bplan.cache).expanduser().resolve()
        self._bplan_input = Path(bplan_path or cfg.ref.bplan.cache).expanduser().resolve()
        if not self._bplan_path.exists():
            raise WeatherClientError(f"BPlan not found: {self._bplan_path}")

        self._cache_dir = Path(cache_dir or cfg.cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_bplan(self) -> Dict[str, Any]:
        if self._bplan_cache.exists():
            return read_cache(self._bplan_cache)
        else:
            return fetch_and_parse_bplan(self._bplan_input)

    def fetch_midas(
        self,
        *,
        years: range,
        tables: list[str] | None = cfg.midas.tables,
        columns_per_table: Dict[str, list[str]] | None = cfg.midas.columns,
        k: int = 3,
        **download_kwargs,
        ) -> pd.DataFrame:

        loc_df = self._load_bplan()


        return download_locations(
            locations=loc_df,
            years=years,
            tables=tables,
            columns_per_table=columns_per_table,
            k=k,
            out_dir=self._cache_dir,
            **download_kwargs,
        )
