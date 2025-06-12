from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator

class MidasCfg(BaseModel):
    version: str
    tables: Dict[str, str]
    columns: Dict[str, List[str]]

    class Config:
        frozen = True


class WeatherSettings(BaseModel):
    cache_dir: Path = Field(alias="cache_dir")
    midas: MidasCfg

    @validator("cache_dir", pre=True)
    def _expand_cache_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    class Config:
        frozen = True
        allow_population_by_field_name = True


class RefCfg(BaseModel):
    api: str
    url: str
    input: Optional[Path]
    cache: Optional[Path]

    # Convert empty strings to None, otherwise expand->Path
    @validator("input", "cache", pre=True)
    def _clean_optional_paths(cls, v: str | Path | None):
        if v in ("", None):
            return None
        return Path(v).expanduser().resolve()

    class Config:
        frozen = True


class Loc2ElrCfg(RefCfg):
    loc_id_field: str = Field(alias="location_code")
    max_distance_m: int = Field(alias="max_distance_m")
    seg_length_mi: int = Field(alias="seg_len_mi")

    class Config:
        frozen = True
        allow_population_by_field_name = True


class Settings(BaseModel):
    weather: WeatherSettings
    timetable: RefCfg
    delay: RefCfg
    loc2elr: Loc2ElrCfg
    ref: Dict[str, RefCfg]

    class Config:
        frozen = True


# ──────────────────────────────────────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────────────────────────────────────
def load_settings(path: str | Path | None = None) -> Settings:
    """
    Load **settings.yaml** and return a validated ``Settings`` object.

    Parameters
    ----------
    path
        Explicit path to YAML.  If *None*, file named ``settings.yaml`` in the
        same directory as this module is used.

    Raises
    ------
    FileNotFoundError
        If YAML file is missing.
    SystemExit
        With JSON payload of all validation errors.
    """
    yaml_path = Path(path or __file__).with_name("settings.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())

    try:
        return Settings.parse_obj(raw)
    except ValidationError as exc:
        # Print all errors at once and abort
        raise SystemExit(exc.json(indent=2))


settings: Settings = load_settings()