# NetRail-Incident-Analysis Package

NetRail-Incident-Analysis is a prototype refactoring of a older rail incident-delay model, currently it ingests raw feeds (e.g. weather, timetables, delay incidents) and produces clean, per-segment, per-hour feature datasets for modelling and analysis.

In future versions, this will include incident and delay modelling subpackages.

---

## Contents
``` 
src/
├── rail_data/
│   ├── io/        # Raw data ingestion, caching, and parsing
│   ├── features/  # Feature engineering on cached data
│   └── models/    # Model training utilities (rough draft!)
```

## Key Concepts

- **Track segments** are identified by `ELR_MIL` codes (Engineer’s Line Reference + milepoint bin).
- All datasets are **hourly resolution**, partitioned by segment and time.
- The pipeline works in **three stages**:
  1. **[Data ingestion (`io/`)](src/rail_data/io/readme.md)**
     Fetches and normalises raw feeds (weather, train schedules, delay logs, holidays, shapefiles…).
  2. **[Feature engineering (`features/`)](src/rail_data/features/readme.md)**
     Builds timebases, aggregates weather, counts trains/incidents, and outputs partitioned Parquet datasets.
  3. **[Modelling (`models/`)](src/rail_data/models/readme.md)**
     Combines feature tables and fits statistical models for incident data.
---

## Example Workflow

```python
import rail_data

rail_data.io.get_datasets("2024-01-01", "2024-12-31")

rail_datafeatures.create_datasets("2024-01-01", "2024-12-31")
