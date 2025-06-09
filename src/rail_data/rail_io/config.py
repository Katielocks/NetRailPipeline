from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import yaml

@dataclass(frozen=True)
class MidasCfg:
    version: str
    tables: Dict[str, str]
    columns: Dict[str, List[str]]

@dataclass(frozen=True)
class WeatherSettings:
    cache_dir: Path
    midas: MidasCfg

@dataclass(frozen=True)
class RefCfg:
      api: str
      url: Path
      input: Path
      cache: Path

@dataclass(frozen=True)
class Settings:
    weather: WeatherSettings
    timetable: RefCfg
    delay: RefCfg
    ref: Dict[str, RefCfg]


def _loadWeatherSettings(weather_dict: dict[str, Any]) -> WeatherSettings:
    if not isinstance(weather_dict, dict):
        raise RuntimeError("Missing or invalid 'weather' section in settings.yaml")

    cache_dir_str = weather_dict.get("cache_dir")
    if not (isinstance(cache_dir_str, str)):
        raise RuntimeError("Missing or invalid 'cache_dir' under weather in settings.yaml")
    cache_dir = Path(cache_dir_str)

    midas_dict = weather_dict.get("midas")
    if not isinstance(midas_dict, dict):
        raise RuntimeError("Missing or invalid 'midas' section under weather in settings.yaml")

    version = midas_dict.get("version")
    tables = midas_dict.get("tables")
    columns = midas_dict.get("columns")

    if not isinstance(version, str):
        raise RuntimeError("Missing or invalid 'version' under weather.midas in settings.yaml")
    if not (isinstance(tables, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in tables.items())):
        raise RuntimeError("Missing or invalid 'tables' mapping under weather.midas in settings.yaml")
    if not (
        isinstance(columns, dict)
        and all(
            isinstance(k, str)
            and isinstance(v, list)
            and all(isinstance(item, str) for item in v)
            for k, v in columns.items()
        )
    ):
        raise RuntimeError("Missing or invalid 'columns' mapping under weather.midas in settings.yaml")

    midas_cfg = MidasCfg(
        version=version,
        tables=tables,
        columns=columns,
    )

    return WeatherSettings(cache_dir=cache_dir, midas=midas_cfg)


def _loadRefSettings(ref_dict: dict[str, Any]) -> Dict[str, RefCfg]:
    if not isinstance(ref_dict, dict):
        raise RuntimeError("Missing or invalid 'ref' section in settings.yaml")

    result: Dict[str, RefCfg] = {}

    for ref_name, block in ref_dict.items():
        if not isinstance(ref_name, str):
            raise RuntimeError(f"Reference name '{ref_name}' must be a string")
        if not isinstance(block, dict):
            raise RuntimeError(f"Reference '{ref_name}' must be a mapping; got {type(block).__name__!r}")
        cfg = _loadRefCfg(ref_name, block)
        result[ref_name] = cfg
    return result

def _loadRefCfg(ref_name: str,block:Dict[str]) -> Dict[str, RefCfg]:
    if not isinstance(block, dict):
            raise RuntimeError(f"Reference '{ref_name}' must be a mapping; got {type(block).__name__!r}")
        
    required_keys = {"api", "url", "input", "cache"}
    missing_keys = required_keys - set(block.keys())
    extra_keys   = set(block.keys()) - required_keys

    if missing_keys:
        raise RuntimeError(f"Reference '{ref_name}' is missing keys: {missing_keys}")
    if extra_keys:
        raise RuntimeError(f"Reference '{ref_name}' has unexpected keys: {extra_keys}")

    api_val     = block["api"]
    url_val     = block["url"]
    input_val   = block["input"]
    cache_val   = block["cache"]

    if not isinstance(api_val, str):
        raise RuntimeError(f"Reference '{ref_name}': 'api' must be a string")
    if not isinstance(url_val, str):
        raise RuntimeError(f"Reference '{ref_name}': 'url' must be a string")
    if not isinstance(input_val, str):
        raise RuntimeError(f"Reference '{ref_name}': 'input' must be a string")
    if not isinstance(cache_val, str):
        raise RuntimeError(f"Reference '{ref_name}': 'cache' must be a string")
    
    url_path   = Path(url_val)
    input_path = Path(input_val)
    cache_path = Path(cache_val)

    cfg = RefCfg(
        api=api_val,
        url=url_path,
        input=input_path,
        cache=cache_path
    )

    return cfg

def _loadSettings() -> Settings:
    base_dir = Path(__file__).parent
    settings_path = base_dir / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Cannot find settings.yaml at {settings_path}")

    raw: dict[str, Any] = yaml.safe_load(settings_path.read_text())
    io_dict = raw.get("io")
    if not isinstance(io_dict, dict):
        raise RuntimeError("Missing or invalid top-level 'io' section in settings.yaml")

    weather_block = io_dict.get("weather")
    weather_settings = _loadWeatherSettings(weather_block)

    ref_block = io_dict.get("ref")
    ref_settings = _loadRefSettings(ref_block)

    timetable_block = io_dict.get("timetable")
    if not isinstance(timetable_block, dict):
        raise RuntimeError("Missing or invalid 'timetable' section in settings.yaml")
    timetable_settings = _loadRefCfg("timetable", timetable_block)
    delay_block = io_dict.get("delay")
    if not isinstance(delay_block, dict):
        raise RuntimeError("Missing or invalid 'delay' section in settings.yaml")
    delay_settings = _loadRefCfg("delay", delay_block)
    if not weather_settings or not timetable_settings or not delay_settings or not ref_settings:
        raise RuntimeError("Missing required sections in settings.yaml")
    
    return Settings(
        weather=weather_settings,
        timetable=timetable_settings,
        delay=delay_settings,
        ref=ref_settings,
    )

settings: Settings = _loadSettings()
