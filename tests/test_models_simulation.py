import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.models.config import load_settings
from rail_data.models.simulation import simulate_glm_counts

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "src" / "rail_data" / "models" / "settings.yaml"


def test_simulation_config():
    cfg = load_settings(SETTINGS_PATH)
    assert cfg.simulation.draws == 100
    assert cfg.simulation.seed == 0


def test_simulate_glm_counts_shape_and_seed():
    X = pd.DataFrame({"x": [0, 1, 2]})
    X = sm.add_constant(X)
    y = pd.Series([1, 3, 5])
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    sims1 = simulate_glm_counts(model, X, n_iter=5, seed=1)
    sims2 = simulate_glm_counts(model, X, n_iter=5, seed=1)
    assert sims1.shape == (5, len(X))
    assert np.array_equal(sims1, sims2)