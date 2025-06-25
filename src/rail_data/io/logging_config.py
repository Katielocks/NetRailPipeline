import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

class RelPathFilter(logging.Filter):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root.resolve()

    def _is_pathlike_str(self, s: str) -> bool:
        return any(sep in s for sep in (os.sep, os.altsep or '')) or ':' in s[:3]

    def _maybe_relpath(self, value: Any) -> Any:
        if isinstance(value, Path):
            p = value.resolve()
        elif isinstance(value, (str, os.PathLike)) and self._is_pathlike_str(str(value)):
            p = Path(value).expanduser().resolve()
        else:
            return value

        try:                               
            rel = p.relative_to(self.project_root)
            return str(rel)
        except ValueError:                 
            return str(p)

    def _convert_args(self, args: Any) -> Any:
        if isinstance(args, Mapping):          
            return {k: self._maybe_relpath(v) for k, v in args.items()}
        if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
            return tuple(self._maybe_relpath(v) for v in args)
        return args


    def filter(self, record: logging.LogRecord) -> bool: 
        if record.args:
            record.args = self._convert_args(record.args)
        return True

def find_project_root_above_src() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name.lower() == "src":
            return parent.parent.resolve()
    return here.parents[-1].resolve()

def setup_logging(level=logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:                    
        return

    project_root = find_project_root_above_src()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.addFilter(RelPathFilter(project_root))

    fmt = "%(asctime)s %(levelname)-5s [%(module)s]: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    root.addHandler(handler)
    root.setLevel(level)
