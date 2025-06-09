from __future__ import annotations
from typing import Callable, Union
import pandas as pd
from pathlib import Path

_OUTPUT_WRITERS: dict[str, Callable[[pd.DataFrame, Path], None]] = {
    "csv":     lambda df, p: df.to_csv(p, index=False),
    "parquet": lambda df, p: df.to_parquet(p, index=False),
    "json":    lambda df, p: df.to_json(p, orient="records"),
}

_INPUT_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "csv":     lambda p: pd.read_csv(p),
    "parquet": lambda p: pd.read_parquet(p),
    "json":    lambda p: pd.read_json(p, orient="records"),
}

def read_cache(cache_path: Union[str, Path]) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file '{cache_path}' does not exist.")
    if not cache_path.is_file():
        raise ValueError(f"Cache path '{cache_path}' is not a file.")

    cache_fmt = cache_path.suffix.lstrip(".").lower()
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
        df = _INPUT_READERS[cache_fmt](cache_path)
    except Exception as e:
        raise IOError(
            f"Failed to read cache file '{cache_path}' as {cache_fmt}: {e}"
        ) from e
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Reader for format '{cache_fmt}' did not return a DataFrame."
        )
    return df

def write_cache(cache_path: Union[str, Path], df: pd.DataFrame, mdir: bool = True) -> None:
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
        _OUTPUT_WRITERS[cache_fmt](df, cache_path)
    except Exception as e:
        raise IOError(
            f"Failed to write DataFrame to '{cache_path}' as {cache_fmt}: {e}"
        ) from e
