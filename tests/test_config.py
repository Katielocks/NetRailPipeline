import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.io.config import load_settings

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "src" / "rail_data" / "io" / "settings.yaml"

def test_load_settings_matches_yaml():
    cfg = load_settings(SETTINGS_PATH)
    raw = yaml.safe_load(SETTINGS_PATH.read_text())
    assert cfg.weather.cache_format == raw["weather"]["cache_format"]
    assert cfg.delay.cache_format == raw["delay"]["cache_format"]
    assert cfg.geospatial.seg_length_mi == raw["geospatial"]["seg_len_mi"]


def test_load_settings_custom_file(tmp_path: Path):
    yaml_path = tmp_path / "settings.yaml"
    raw = yaml.safe_load(SETTINGS_PATH.read_text())
    raw["weather"]["cache_format"] = "parquet"
    raw["weather"]["cache_dir"] = "~/temp_weather"
    yaml_path.write_text(yaml.safe_dump(raw))
    cfg = load_settings(yaml_path)
    assert cfg.weather.cache_format == "parquet"
    assert cfg.weather.cache_dir == Path("~/temp_weather").expanduser().resolve()