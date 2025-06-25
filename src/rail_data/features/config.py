from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Mapping

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    RootModel,
    ConfigDict,
)


class TableFeature(BaseModel):
    action: str
    window_hours: int

    model_config = ConfigDict(frozen=True)


class ColumnFeatureMap(RootModel[Dict[str, TableFeature]]):
    model_config = ConfigDict(frozen=True)

    @property
    def _data(self) -> Dict[str, TableFeature]:
        return self.root

    def __getattr__(self, item: str) -> TableFeature:
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> TableFeature:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def items(self) -> Mapping[str, TableFeature].items:
        return self._data.items()


class TableFeatureMap(RootModel[Dict[str, ColumnFeatureMap]]):
    model_config = ConfigDict(frozen=True)

    @property
    def _data(self) -> Dict[str, ColumnFeatureMap]:
        return self.root

    def __getattr__(self, item: str) -> ColumnFeatureMap:
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> ColumnFeatureMap:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def items(self) -> Mapping[str, ColumnFeatureMap].items:
        return self._data.items()


class FlagCfg(BaseModel):
    table: TableFeatureMap
    threshold: float

    model_config = ConfigDict(frozen=True)


class FlagCfgMap(RootModel[Dict[str, FlagCfg]]):
    model_config = ConfigDict(frozen=True)

    @property
    def _data(self) -> Dict[str, FlagCfg]:
        return self.root

    def __getattr__(self, item: str) -> FlagCfg:
        try:
            return self._data[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __getitem__(self, item: str) -> FlagCfg:
        return self._data[item]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def items(self) -> Mapping[str, FlagCfg].items:
        return self._data.items()


class WeatherFeatures(BaseModel):
    tables: TableFeatureMap
    flags: FlagCfgMap

    model_config = ConfigDict(frozen=True)


class WeatherCfg(BaseModel):
    features: WeatherFeatures
    cache_dir: Path = Field(alias = "cache_dir")
    cache_format: Path = Field(alias = "cache_format")
    parquet_dir: Path = Field(alias="parquet_dir")

    @field_validator("parquet_dir", mode="before")
    def _expand_parquet_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    model_config = ConfigDict(frozen=True, validate_by_name=True)


class _ParquetCfg(BaseModel):
    parquet_dir: Path = Field(alias="parquet_dir")

    @field_validator("parquet_dir", mode="before")
    def _expand(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    model_config = ConfigDict(frozen=True, validate_by_name=True)



class Settings(BaseModel):
    weather: WeatherCfg
    train_counts: _ParquetCfg
    incidents: _ParquetCfg
    main: _ParquetCfg

    model_config = ConfigDict(frozen=True)


def load_settings(path: str | Path | None = None) -> Settings:
    """Load *settings.yaml* and return a fully validated :class:`Settings`."""

    yaml_path = Path(path or __file__).with_name("settings.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(exc.json(indent=2))


settings: Settings = load_settings()