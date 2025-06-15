from __future__ import annotations

from session import Session
import pandas as pd
import logging
from .config import settings
from .utils import write_cache

class UKHolidaysClientError(RuntimeError):
    pass

log = logging.getLogger(__name__)
cfg = settings.ref.bank_holiday

def fetch_public_holidays():
    url = cfg.url
    output_path = cfg.cache
    try:
        with Session(token="dummy_token") as ses:
            txt = ses.get_json(
                url
            )
    except Exception as exc:
        raise UKHolidaysClientError(str(exc)) from exc
    df = pd.read_json(txt)
    if output_path:
        write_cache(output_path,df)
        return df
    else: 
        return df
