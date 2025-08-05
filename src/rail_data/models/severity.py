from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[Tuple[float, float]], np.ndarray]


def sample_incident_durations(
    posterior_samples: ArrayLike,
    n_incidents: int = 1,
    rng: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """Draw delay durations for simulated incidents.

    Parameters
    ----------
    posterior_samples:
        Sequence or ``(n, 2)`` array containing posterior samples of the
        Weibull parameters ``(k, lambda)``.  Each row represents one draw
        from the posterior distribution.
    n_incidents:
        Number of incidents to simulate.  A distinct duration is sampled
        for each incident.
    rng:
        Optional random number generator or seed passed to
        :func:`numpy.random.default_rng`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_incidents,)`` with sampled delay durations in
        minutes.

    Notes
    -----
    The sampling procedure incorporates parameter uncertainty by
    selecting a random posterior sample for each incident before drawing
    from the corresponding Weibull distribution.
    """

    rng = np.random.default_rng(rng)
    samples = np.asarray(posterior_samples, dtype=float)

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            "posterior_samples must be an array-like of shape (n, 2) with columns (k, lambda)"
        )
    if n_incidents < 1:
        raise ValueError("n_incidents must be at least 1")

    idx = rng.integers(0, samples.shape[0], size=n_incidents)
    ks = samples[idx, 0]
    lambdas = samples[idx, 1]
    durations = rng.weibull(ks) * lambdas
    return durations


__all__ = ["sample_incident_durations"]
