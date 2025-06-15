from __future__ import annotations
from typing import Callable, Union
import pandas as pd
from pathlib import Path
import logging
log = logging.getLogger(__name__)

_OUTPUT_WRITERS: dict[str, Callable[[pd.DataFrame, Path, str | None], None]] = {
    "csv":     lambda df, p, comp=None:     df.to_csv(p, index=False, compression=comp or "infer"),
    "parquet": lambda df, p, comp=None:     df.to_parquet(p, index=False),
    "json":    lambda df, p, comp=None:     df.to_json(p, orient="records", compression=comp or "infer"),
}

_INPUT_READERS: dict[str, Callable[[Path, str | None], pd.DataFrame]] = {
    "csv":     lambda p, comp=None:        pd.read_csv(p, compression=comp or "infer"),
    "parquet": lambda p, comp=None:        pd.read_parquet(p),
    "json":    lambda p, comp=None:        pd.read_json(p, orient="records", compression=comp or "infer"),
}
_COMP_EXTS = {".gz": "gzip",
              ".gzip": "gzip",
              ".bz2": "bz2",
              ".xz": "xz",
              ".zip": "zip"}         


def _detect_format_and_compression(path: Path) -> tuple[str, str | None]:
    """
    Returns (<format>, <compression>) where <compression> is a pandas
    """
    suffixes = path.suffixes                                  
    if not suffixes:
        raise ValueError(f"No extension found for '{path}'.")
    compression = None
    if suffixes[-1] in _COMP_EXTS:
        compression = _COMP_EXTS[suffixes[-1]]
        if len(suffixes) < 2:                               
            raise ValueError(f"Compressed file '{path}' needs an inner extension "
                             "(eg. *.csv.gz, *.json.bz2).")
        fmt = suffixes[-2].lstrip(".").lower()
    else:
        fmt = suffixes[-1].lstrip(".").lower()
    return fmt, compression


def read_cache(cache_path: Union[str, Path]) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file '{cache_path}' does not exist.")
    if not cache_path.is_file():
        raise ValueError(f"Cache path '{cache_path}' is not a file.")

    cache_fmt,comp = _detect_format_and_compression(cache_path)
    if not cache_fmt:
        raise ValueError(
            f"No file extension found for '{cache_path}'."
            f" Choose one of {list(_INPUT_READERS.keys())}."
        )
    if cache_fmt not in _INPUT_READERS:
        raise ValueError(
            f"Unsupported input format '{cache_fmt}'."
            f" Supported formats are: {list(_INPUT_READERS.keys())}."
        )
    try:
        df = _INPUT_READERS[cache_fmt](cache_path,comp)
    except Exception as e:
        raise IOError(
            f"Failed to read cache file '{cache_path}' as {cache_fmt}: {e}"
        ) from e
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Reader for format '{cache_fmt}' did not return a DataFrame."
        )
    return df

def write_cache(cache_path: Union[str, Path], df: pd.DataFrame, mdir: bool = True,comp = None) -> None:
    cache_path = Path(cache_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame to write, got {type(df)}."
        )
    parent = cache_path.parent
    if mdir:
        parent.mkdir(parents=True, exist_ok=True)
    else:
        if not parent.exists():
            raise FileNotFoundError(f"Directory '{parent}' does not exist.")
        if not parent.is_dir():
            raise ValueError(f"Parent path '{parent}' is not a directory.")
    cache_fmt = cache_path.suffix.lstrip(".").lower()
    if not cache_fmt:
        raise ValueError(
            f"No file extension found for '{cache_path}'."
            f" Choose one of {list(_OUTPUT_WRITERS.keys())}."
        )
    if cache_fmt not in _OUTPUT_WRITERS:
        raise ValueError(
            f"Unsupported output format '{cache_fmt}'."
            f" Supported formats are: {list(_OUTPUT_WRITERS.keys())}."
        )
    try:
        _OUTPUT_WRITERS[cache_fmt](df, cache_path,comp)
    except Exception as e:
        raise IOError(
            f"Failed to write DataFrame to '{cache_path}' as {cache_fmt}: {e}"
        ) from e


def get_cache(
    cache_path: Union[str, Path],
    input_path: Union[str, Path, None] = None,
    gen_func: Callable | None = None
):
    cache_path = Path(cache_path)
    input_path = Path(input_path) if input_path is not None else None

    if not cache_path.parent.is_dir():
        raise ValueError(f"{cache_path.parent!r} is not a directory")

    candidates = [
        f".{p.suffix}"
        for p in cache_path.parent.iterdir()
        if p.is_file()
        and p.stem == cache_path.stem
        and p.suffix != cache_path.suffix
        and p.suffix in _INPUT_READERS
    ]
    if cache_path.exists():
        return read_cache(cache_path)
    if input_path and input_path.exists() and gen_func and isinstance(gen_func,Callable):
        return gen_func(input_path, cache_path)

    msg = f"No cache file found at {cache_path!r}"
    if input_path:
        msg += f" and input file {input_path!r} does not exist"
        msg += "."
    if gen_func:
        msg += f" and input function {gen_func!r} does not exist"
        msg += "."
    if candidates:
        msg += f" Did you mean one of these formats? {', '.join(candidates)}"
    raise FileNotFoundError(msg)
