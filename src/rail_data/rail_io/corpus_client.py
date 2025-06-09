from pathlib import Path
from typing import Iterator, Union
import gzip
import json
import io
import logging
import zipfile
import pandas as pd
from utils import read_cache,write_cache
from config import settings


class CORPUSClientError(Exception):
    """Base error for corpus I/O."""

DEFAULT_ENCODING = "latin-1"
log = logging.getLogger(__name__)
cfg = settings.ref.corpus

def _open_file(file_path: Union[str, Path], encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffixes = p.suffixes
    is_gz = suffixes and suffixes[-1] == ".gz"

    fmt = (suffixes[-2] if is_gz and len(suffixes) > 1 else suffixes[-1]).lstrip('.')


    if fmt == "json":
        if is_gz:
            with gzip.open(p, "rb") as gz_fh:
                text = gz_fh.read().decode(encoding)
        else:
            text = p.read_text(encoding=encoding)
        data = json.loads(text)
        if isinstance(data, dict) and "TIPLOCDATA" in data:
            data = data["TIPLOCDATA"]
        elif isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        return pd.json_normalize(data)
    else:
        if is_gz:
            with gzip.open(p, "rb") as gz_fh:
                raw = gz_fh.read()
            buffer = io.BytesIO(raw)
            return read_cache(buffer)
        return read_cache(p)

    


def get_corpus():
    input_path: str | Path | None = cfg.input
    if not input_path:
        raise CORPUSClientError("No input archive specified in settings")

    output_path: str | Path | None = cfg.cache
    df = _open_file(input_path)
    write_cache(output_path,df)
    return df