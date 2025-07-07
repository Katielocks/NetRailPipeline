# `rail_data.models` — Overview

The **models** subpackage contains helpers for assembling modelling
frames from feature tables and fitting statistical models with
`statsmodels`.

---

## Quick start

```python
from rail_data.models import build_modelling_frame, train_first_elr_model

# Merge feature tables for a single track segment
frame = build_modelling_frame(elr_mil="A1231")

# Fit models for the first available segment
results = train_first_elr_model()