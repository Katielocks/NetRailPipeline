from pathlib import Path
from typing import Union
import logging
import pandas as pd
import json
from .utils import write_cache,read_cache,get_cache       
from .config import settings

class CORPUSClientError(Exception):
    """Base error for corpus I/O."""

DEFAULT_ENCODING = "latin-1"
log = logging.getLogger(__name__)


def extract_corpus(file_path: Union[str, Path],
                   cache_path: Union[str, Path] = None,
                   encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    """Load a CORPUS file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    file_path:
        Path to the CORPUS file.  Can be a plain JSON/CSV file or a
        gzip-compressed variant.
    encoding:
        Text encoding used for JSON files.  Defaults to ``"latin-1"`` which is
        the format most CORPUS exports use.

    Returns
    -------
    pandas.DataFrame
        Parsed CORPUS records.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffixes = p.suffixes
    is_gz = suffixes and suffixes[-1] == ".gz"
    fmt = (suffixes[-2] if is_gz else suffixes[-1]).lstrip('.').lower()
    if p and p.exists():
        if fmt == "json":
            df_like = pd.read_json(p,
                                encoding=encoding,
                                compression="gzip" if is_gz else None)
            if isinstance(df_like, pd.Series) or (isinstance(df_like, pd.DataFrame)
                                                and df_like.columns.size == 1):
                data = df_like.iloc[0] if isinstance(df_like, pd.Series) else df_like.iloc[:, 0]
                try:
                    as_dict = json.loads(data) if isinstance(data, str) else data
                    if isinstance(as_dict, dict) and "TIPLOCDATA" in as_dict:
                        df = pd.json_normalize(as_dict["TIPLOCDATA"])
                    elif isinstance(as_dict, dict) and len(as_dict) == 1:
                        df = pd.json_normalize(next(iter(as_dict.values())))
                    else:
                        df = pd.json_normalize(as_dict)
                except Exception:
                    df = df_like 
            else:
                df = df_like
        else:
            df =  read_cache(p)
    if cache_path and Path(cache_path).exists():
                    write_cache(cache_path,df)
    else: 
       FileNotFoundError(f"No file found at {p!r}")


def get_corpus(cache_path: Union[str, Path],input_path = Union[str, Path]) -> pd.DataFrame:
    """Return the CORPUS DataFrame from *cache_path* or generate it.

    The function first checks ``cache_path`` for an existing cached dataset.
    If not found, :func:`extract_corpus` is called with ``input_path`` and the
    resulting frame is cached.

    Parameters
    ----------
    cache_path
        Expected location of the cached DataFrame.
    input_path
        Path to the raw CORPUS source file used when the cache is missing.

    Returns
    -------
    pandas.DataFrame
        The CORPUS dataset.
    """
        
    if settings and settings.ref.corpus:
        cache_path = cache_path or settings.ref.corpus.cache
        input_path = input_path or settings.ref.corpus.input
    return get_cache(cache_path,input_path,extract_corpus)
