from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLMResults

from .config import settings


def simulate_glm_counts(
    result: GLMResults,
    X: pd.DataFrame,
    n_iter: int | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Run a simple Monte Carlo simulation for ``result``.

    Parameters
    ----------
    result:
        Fitted :class:`~statsmodels.genmod.generalized_linear_model.GLMResults`.
    X:
        Design matrix used for prediction.  Must contain the same columns as
        were used for fitting ``result``.
    n_iter:
        Number of simulation iterations.  Defaults to
        ``settings.simulation.draws`` if ``None``.
    seed:
        Optional random seed.  If not provided, ``settings.simulation.seed``
        is used.

    Returns
    -------
    numpy.ndarray
        Simulated counts with shape ``(n_iter, len(X))``.
    """

    n_iter = n_iter or settings.simulation.draws
    seed = settings.simulation.seed if seed is None else seed

    rng = np.random.default_rng(seed)

    X_mat = np.asarray(X)
    # Draw coefficients from the estimated parameter distribution
    coef_draws = rng.multivariate_normal(
        mean=result.params, cov=result.cov_params(), size=n_iter
    )

    linear_pred = coef_draws @ X_mat.T
    mu = result.family.link.inverse(linear_pred)
    return rng.poisson(mu)