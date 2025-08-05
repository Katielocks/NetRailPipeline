import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rail_data.models.severity import sample_incident_durations


def test_sample_incident_durations_reproducible():
    posterior = np.array([[1.0, 2.0], [2.0, 3.0], [1.5, 4.0]])
    rng = np.random.default_rng(42)
    durations = sample_incident_durations(posterior, n_incidents=5, rng=rng)
    expected = np.array(
        [0.55958858, 0.78197785, 3.61579101, 3.56225297, 5.30270342]
    )
    assert durations.shape == (5,)
    assert np.allclose(durations, expected)


def test_sample_incident_durations_invalid_shape():
    posterior = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        sample_incident_durations(posterior)