"""Convenient access to model-training helpers."""

from .config import settings
from .severity import sample_incident_durations

def build_modelling_frame(*args, **kwargs):
    """Lazy wrapper for :func:`construct_frame.build_modelling_frame`."""
    from .construct_frame import build_modelling_frame as _impl

    return _impl(*args, **kwargs)


def build_formula(*args, **kwargs):
    """Lazy wrapper for :func:`modelling.build_formula`."""
    from .modelling import build_formula as _impl

    return _impl(*args, **kwargs)


def split_xy(*args, **kwargs):
    """Lazy wrapper for :func:`modelling.split_xy`."""
    from .modelling import split_xy as _impl

    return _impl(*args, **kwargs)


def train_model_for_elr(*args, **kwargs):
    """Lazy wrapper for :func:`modelling.train_model_for_elr`."""
    from .modelling import train_model_for_elr as _impl

    return _impl(*args, **kwargs)


def train_first_elr_model(*args, **kwargs):
    """Lazy wrapper for :func:`modelling.train_first_elr_model`."""
    from .modelling import train_first_elr_model as _impl

    return _impl(*args, **kwargs)

def sample_delay_severity(*args, **kwargs):
    """Lazy wrapper for :func:`severity.sample_delay_severity`."""
    from .severity import sample_delay_severity as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "settings",
    "build_modelling_frame",
    "build_formula",
    "split_xy",
    "train_model_for_elr",
    "train_first_elr_model",
    "simulate_glm_counts",
    "sample_incident_durations",
]