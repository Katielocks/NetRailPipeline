from __future__ import annotations

import pandas as pd
import logging
from .session import Session
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
            data = ses.get_json(
                url
            )
    except Exception as exc:
        raise UKHolidaysClientError(str(exc)) from exc
    events = data.get("england-and-wales", {}).get("events", [])
    df = pd.json_normalize(events)
    if output_path:
        write_cache(output_path, df)
        log.info("Saved holiday data to %s", output_path)
        return df
    else:
        return df
