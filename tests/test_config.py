import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rail_data", "rail_io"))

from config import load_settings


def test_load_settings_defaults():
    cfg = load_settings()
    assert cfg.weather.cache_format == "csv"
    assert cfg.geospatial.seg_length_mi == 8