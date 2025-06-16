import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.rail_io.config import load_settings


def test_load_settings_defaults():
    cfg = load_settings()
    assert cfg.weather.cache_format == "csv"
    assert cfg.geospatial.seg_length_mi == 8