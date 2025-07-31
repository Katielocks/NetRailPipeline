
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, List, Iterable, Tuple
import glob
import logging
import re
import numpy as np
import duckdb
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_integer_dtype,
)
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import statsmodels.formula.api as smf
import statsmodels.api as sm

from ..features.config import settings as feature_settings  

from .construct_frame import build_modelling_frame,_first_elr_mil

log = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*") 

def _quote_if_needed(col: str) -> str:
    """Wrap *col* in Q("…") if Patsy needs it."""
    return col if _IDENTIFIER_RE.fullmatch(col) else f'Q("{col}")'

def categorize_columns(
    df: pd.DataFrame,
    *,
    response: str = "y",
    cat_unique_cutoff: int = 20,
    force_numeric: Iterable[str] | None = ("train_count",),
    ) -> dict[str, list[str]]:
    """Return columns grouped by modelling data type."""

    force_numeric = set(force_numeric or [])
    categories = {"numeric": [], "categorical": []}

    for col in df.columns:
        if col == response:
            continue

        escaped = _quote_if_needed(col)
        s = df[col]

        if col in force_numeric:
            categories["numeric"].append(col)
            continue

        if (
            is_object_dtype(s)
            or is_string_dtype(s)
            or is_categorical_dtype(s)
            or (is_integer_dtype(s) and s.nunique(dropna=False) <= cat_unique_cutoff)
        ):
            categories["categorical"].append(col)
        else:
            categories["numeric"].append(col)

    return categories


def build_formula(
    df: pd.DataFrame,
    *,
    response: str = "y",
    cat_unique_cutoff: int = 20,
    force_numeric: Iterable[str] | None = ("train_count",),
) -> str:
    """Build a Patsy model formula for ``df``."""

    categories = categorize_columns(
        df,
        response=response,
        cat_unique_cutoff=cat_unique_cutoff,
        force_numeric=force_numeric,
    )

    terms: list[str] = []
    for col in sorted(categories["numeric"]):
        terms.append(_quote_if_needed(col))
    for col in sorted(categories["categorical"]):
        terms.append(f"C({_quote_if_needed(col)})")

    return f"{response} ~ " + " + ".join(terms)


def split_xy(
    df: pd.DataFrame,
    *,
    incident_prefix: str = "INCIDENT_",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return predictors **X** and target matrix **Y**.

    Y = all columns whose names start with *incident_prefix*,
    X = everything else.
    """
    incident_cols = [c for c in df.columns if c.startswith(incident_prefix)]
    if not incident_cols:
        raise ValueError("No incident columns found; check prefix or data sources.")

    y = df[incident_cols].copy()
    X = df.drop(columns=incident_cols).copy()
    return X, y


def train_model_for_elr_zinb(
    elr_mil: str,
    *,
    time_filter: Optional[Dict[str, Union[int, List[int]]]] = None,
    inflation_rhs: str | None = "1",
    dist: str = "negbin",
    method: str = "bfgs",
    maxiter: int = 1000,
    return_xy: bool = False,
):
    df = build_modelling_frame(elr_mil=elr_mil, time_filter=time_filter)
    df.drop(columns=["hour", "day", "month", "year", "run_hour"], errors="ignore", inplace=True)
    X, Y = split_xy(df)
    if return_xy:
        return X, Y
    results: Dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}
    for col in Y.columns:
        y = Y[col].astype(float)
        if y.sum() == 0:
            continue
        df_model = X.copy()
        df_model["y"] = y
        rhs = build_formula(df_model).split("~", 1)[1].strip()
        inflation_part = rhs if inflation_rhs is None else inflation_rhs
        formula = f"y ~ {rhs} | {inflation_part}"
        fit = ZeroInflatedNegativeBinomialP(
            endog=y.values,
            exog= X.values,
            exog_infl=X.values,   
            inflation="logit"
        )
        try:
            fit = ZeroInflatedNegativeBinomialP(
            endog=y.values,
            exog= X.values,
            exog_infl=X.values,   
            inflation="logit"
            )
            results[col] = fit
            log.info(
                "Fitted ZINB for %s (alpha=%.3g, LL=%.1f)",
                col,
                fit.params.get("alpha", np.nan),
                fit.llf,
            )
        except Exception:
            log.exception("Failed to fit ZINB for %s", col)
    if not results:
        raise RuntimeError("No incident models were fitted successfully.")
    first_key = next(iter(results))
    print(results[first_key].summary().as_text())
    return results

def train_model_for_elr(
    elr_mil: str,
    *,
    time_filter: Optional[Dict[str, Union[int, List[int]]]] = None,
    return_xy: bool = False,
):
    df = build_modelling_frame(elr_mil=elr_mil, time_filter=time_filter)
    df.drop(columns=["hour", "day", "month", "year", "run_hour"], errors="ignore", inplace=True)
    X, Y = split_xy(df)

    if return_xy:
        return X, Y

    X_enc = (
        pd.get_dummies(X, drop_first=True)  
        .astype(float)
    )
    X_enc = sm.add_constant(X_enc, has_constant="add") 

    mask = np.isfinite(X_enc).all(axis=1)
    X_enc = X_enc.loc[mask]
    Y     = Y.loc[mask]        
    results: dict[str, sm.GLMResults] = {}

    for col in Y.columns:
        y = Y[col].astype(float)

        if y.sum() == 0:
            continue

        try:
            results[col] = sm.GLM(
                y,
                X_enc,
                family=sm.families.Poisson()
            ).fit()
        except ValueError as err:
            pass

    print("Fitted:", list(results))
    print(results[list(results)[0]].summary().as_text())
    return results


def train_first_elr_model(*, return_xy: bool = False):
    """Convenience wrapper for the first ELR_MIL partition found on disk."""
    elr_mil = _first_elr_mil(feature_settings.incidents.parquet_dir)
    return train_model_for_elr_zinb(elr_mil, return_xy=return_xy)
