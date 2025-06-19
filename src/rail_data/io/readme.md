# `rail_data.io` — Overview

The **io** sub-package is the one-stop shop for **ingesting, cleaning, and caching** all raw railway data (weather, timetables, delay logs, geospatial, reference tables, …).  
Everything is cached locally so that downstream code can work offline and fast.

---

## Directory layout

| File | What it does             |
| ---- | ------------------------ |
| `main.py` | **Orchestrators** – `extract_datasets()` (force re-download) and `get_datasets()` (download missing only). |
| `config.py` | Loads `settings.yaml` into a Pydantic `settings` object (all paths, parameters). |
| `location_client.py` | Parses Network Rail **location reference** files (STANOX, TIPLOC…). |
| `track_client.py` | Opens the track-model ZIP; yields the shapefile for GeoPandas. |
| `geospatial_extractor.py` | Maps locations to track segments (`ELR_MIL`), writes `geospatial.parquet`. |
| `weather_extractor.py` | Downloads / filters MIDAS weather tables by year → `archive/weather/*.csv`. |
| `timetable_extractor.py` | Converts CIF timetables to a flat CSV of train “hops”. |
| `delay_extractor.py` | Unzips & normalises **delay logs** into per-period CSVs. |
| `uk_holidays_client.py` | Fetches UK public-holiday dates. |
| `national_rail_client.py` | Thin wrapper around National Rail Data Feeds (authenticated `requests.Session`). |
| `utils.py` | Tiny helpers: `read_cache`, `write_cache`, `get_cache`. |

---

## Quick start

```python
from rail_data import io

io.get_datasets("2024-01-01", "2024-12-31")