import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from rail_data.rail_io import utils


def test_detect_format_simple():
    fmt, comp = utils._detect_format_and_compression(Path("data.csv"))
    assert fmt == "csv"
    assert comp is None


def test_detect_format_compressed():
    fmt, comp = utils._detect_format_and_compression(Path("data.json.gz"))
    assert fmt == "json"
    assert comp == "gzip"


def test_write_read_cache_roundtrip_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "df.csv"
    utils.write_cache(path, df)
    df2 = utils.read_cache(path)
    pd.testing.assert_frame_equal(df, df2)


def test_write_cache_invalid_df(tmp_path):
    with pytest.raises(TypeError):
        utils.write_cache(tmp_path / "bad.csv", [1, 2, 3])


def test_read_cache_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.read_cache(tmp_path / "missing.csv")


def test_get_cache_generate(tmp_path):
    input_file = tmp_path / "input.csv"
    df = pd.DataFrame({"x": [1]})
    df.to_csv(input_file, index=False)
    cache_file = tmp_path / "cache.csv"

    def gen(inp, out):
        d = pd.read_csv(inp)
        utils.write_cache(out, d)
        return d

    result = utils.get_cache(cache_file, input_file, gen)
    pd.testing.assert_frame_equal(result, df)
    assert cache_file.exists()