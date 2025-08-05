# `rail_data.models` — Overview

The **models** sub-package assembles feature tables into modelling frames,
fits statistical or machine-learning models, and supports Monte Carlo
simulation of incident counts and severities.

---

## Directory layout

| File | What it does |
| ---- | ------------ |
| `construct_frame.py` | Merge feature tables into a single modelling frame. |
| `modelling.py` | Build Patsy formulas and train GLM/ZINB models. |
| `simulation.py` | Monte Carlo simulation of incident counts from fitted GLMs. |
| `severity.py` | Sample incident delay durations from Weibull posteriors. |
| `XGBoost.py` | Baseline multi-label classifier using XGBoost. |
| `config.py` | Load `settings.yaml` (paths, training windows, simulation settings). |

---

## Quick start

```python
from rail_data.models.simulation import simulate_glm_counts

# Merge feature tables for a single track segment
frame = build_modelling_frame(elr_mil="A1231")

# Fit models for the first available segment
# Fit a statistical model on the first available segment
result = train_first_elr_model()

# Draw incident-count trajectories via Monte Carlo
X = frame.drop(columns=[c for c in frame.columns if c.startswith("INCIDENT_")])
counts = simulate_glm_counts(result, X)
```