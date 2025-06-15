from __future__ import annotations
import logging
import tempfile
import zipfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from config import settings

log = logging.getLogger(__name__)

class ELRClientError(Exception):
    pass

class ZipFileNotFoundError(ELRClientError):
    pass


@contextmanager
def _open_zip(input_path: Union[str, Path]):
    zip_path = Path(input_path).expanduser().resolve()
    if not zip_path.exists():
        raise ZipFileNotFoundError(f"Local Track Model not found at {zip_path}")
    if zip_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(zip_path, "r") as zf:
            stem = "nwr_trackcentrelines"
            required = {ext: None for ext in (".shp", ".shx", ".dbf")}
            for m in zf.namelist():
                p = Path(m)
                if p.stem.lower() == stem and p.suffix.lower() in required:
                    required[p.suffix.lower()] = m
            if not all(required.values()):
                missing = [ext for ext, path in required.items() if path is None]
                raise ELRClientError(f"Missing {', '.join(missing)} for {stem} in {zip_path.name}")
            temp_dir = Path(tempfile.mkdtemp(prefix="NWR_TM_")).resolve()
            for member in required.values():
                dest = (temp_dir / member).resolve()
                if not str(dest).startswith(str(temp_dir)):
                    raise ELRClientError(f"Unsafe zip member path: {member}")
                zf.extract(member, path=temp_dir)
        try:
            yield temp_dir / f"{stem}.shp"
        finally:
            shutil.rmtree(temp_dir)
    else:
        yield zip_path

def _validate_standalone_shp(path: Path) -> Path:
    if not (path.suffix.lower() == ".shp" and path.is_file()):
        raise FileNotFoundError(f"{path} is not a .shp file")
    stem, parent = path.stem, path.parent
    for ext in (".shx", ".dbf"):
        sib = parent / f"{stem}{ext}"
        if not sib.exists():
            raise ELRClientError(f"Missing {ext} for {stem} in {parent}")
    return path

@contextmanager
def get_track(input_path: Union[str, Path]):
    """Yield the path to a track shapefile from ``input_path``.

    ``input_path`` may be a directory containing the track model, a direct path
    to ``nwr_trackcentrelines.shp`` or a ``.zip`` archive containing the
    shapefile and its ancillary files.  The function yields the resolved path to
    the ``.shp`` file and cleans up any temporary extraction directory when
    used as a context manager.
    """
    input_path = Path(input_path).expanduser().resolve()

    if settings and settings.ref.track_model:
        input_path = input_path or settings.ref.track_model.input

    if input_path.suffix.lower() == ".zip":
        with _open_zip(input_path) as shp:
            yield shp

    elif input_path.suffix.lower() == ".shp":
        shp = _validate_standalone_shp(input_path)
        yield shp                               

    elif input_path.is_dir():
        stem = "nwr_trackcentrelines"
        shp = _validate_standalone_shp(input_path / f"{stem}.shp")
        yield shp                                  
    else:
        raise FileNotFoundError(f"ELR source not found at {input_path}")