from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Iterator, Mapping

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,     
    RootModel,
    ConfigDict,           
)



class MidasCfg(BaseModel):
    version: str
    tables: Dict[str, List[str | None]]

    model_config = ConfigDict(frozen=True)


class WeatherSettings(BaseModel):
    cache_dir: Path = Field(alias="cache_dir")
    cache_format: str = Field(alias="cache_format")
    midas: MidasCfg

    @field_validator("cache_dir", mode="before")
    def _expand_cache_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    model_config = ConfigDict(
        frozen=True,
        validate_by_name=True,   
    )


class RefCfg(BaseModel):
    api: str
    url: str
    input: Optional[Path]
    cache: Optional[Path]

    @field_validator("input", "cache", mode="before")
    def _clean_optional_paths(cls, v: str | Path | None):
        if v in ("", None):
            return None
        return Path(v).expanduser().resolve()

    model_config = ConfigDict(frozen=True)



class RefCfgMap(RootModel[Dict[str, RefCfg]]):
    model_config = ConfigDict(frozen=True)

    @property
    def _data(self) -> Dict[str, RefCfg]:
        return self.root

    def __getattr__(self, item: str) -> RefCfg:   
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> RefCfg:       
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def items(self) -> Mapping[str, RefCfg].items:
        return self._data.items()


class GeospatialCfg(RefCfg):
    loc_id_field: str = Field(alias="location_code")
    max_distance_m: int = Field(alias="max_distance_m")
    seg_length_mi: int = Field(alias="seg_len_mi")

    model_config = ConfigDict(
        frozen=True,
        validate_by_name=True,     # keep alias population behaviour
    )


class Settings(BaseModel):
    weather: WeatherSettings
    timetable: RefCfg
    delay: RefCfg
    geospatial: GeospatialCfg
    ref: RefCfgMap

    model_config = ConfigDict(frozen=True)



def load_settings(path: str | Path | None = None) -> Settings:
    """
    Load **settings.yaml** and return a validated ``Settings`` object.
    """
    yaml_path = Path(path or __file__).with_name("settings.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(exc.json(indent=2))
    

settings: Settings = load_settings()