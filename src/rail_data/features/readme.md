# `rail_data.features` — Overview

The **features** sub-package turns the raw data prepared by `rail_data.io` into tidy, analysis-ready feature tables.  
Everything is pure Python + DuckDB + Pandas; outputs are partitioned Parquet files (`ELR_MIL / year / month / day / hour`).

---

## Directory layout

| File                         | What it does                                                                                  |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| `main.py`                    | **Orchestrator** – `create_datasets(start, end)` calls every other module in the right order. |
| `generate_database.py`       | Builds the **hourly time-base** for every track segment (cyclical time features included).     |
| `convert_weather.py`         | Converts raw MIDAS weather tables to per-segment hourly weather records.                      |
| `sql_weather.py`             | Uses DuckDB to add rolling aggregates / flags (e.g. 48-h min temp, ‘freeze’ flag).            |
| `streaming_train_counts.py`  | Expands the timetable into hourly **train-count** features per segment.                      |
| `extract_incidents.py`       | Summarises delay logs into hourly **incident count** columns (`INCIDENT_*`).                  |
| `utils.py`                   | Common helpers – datetime split, STANOX to `ELR_MIL` mapping, Parquet writer.                   |
| `config.py`                  | Loads `settings.yaml` into a Pydantic `settings` object (all paths, parameters).             |

---

## Quick start

```python
from rail_data.features import create_datasets

# Generate **all** feature tables for January 2025
create_datasets("2025-01-01", "2025-01-31")
