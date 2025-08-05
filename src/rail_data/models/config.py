from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, ConfigDict


class DatasetCfg(BaseModel):
    parquet: Path = Field(alias="parquet")

    model_config = ConfigDict(frozen=True, validate_by_name=True)


class ModelCfg(BaseModel):
    type: str
    formula: str
    offset: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class TrainCfg(BaseModel):
    start: str
    end: str

    model_config = ConfigDict(frozen=True)

class SimulationCfg(BaseModel):
    draws: int = Field(alias="draws")
    seed: Optional[int] = Field(default=None, alias="seed")

    model_config = ConfigDict(frozen=True, validate_by_name=True)


class Settings(BaseModel):
    dataset: DatasetCfg
    model: ModelCfg
    output_dir: Path = Field(alias="output_dir")
    train: TrainCfg
    test: TrainCfg
    simulation: SimulationCfg
    model_config = ConfigDict(frozen=True, validate_by_name=True)


def load_settings(path: str | Path | None = None) -> Settings:
    yaml_path = Path(path or __file__).with_name("settings.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text())

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(exc.json(indent=2))


settings: Settings = load_settings()