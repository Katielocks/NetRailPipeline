from __future__ import annotations

from pathlib import Path
import gc
import logging
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier, DMatrix, train as xgb_train


from .construct_frame import build_modelling_frame
from ..features.config import settings as feature_settings

PARQUET_DIR = Path(feature_settings.incidents.parquet_dir)
INCIDENT_PREFIX = "INCIDENT_"
MODEL_OUT = Path("xgb_incidents_multioutput.pkl")
RANDOM_SEED = 42

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _available_elrs(parquet_dir: Path = PARQUET_DIR) -> List[str]:
    """Return all ELR_MIL partition names found on disk."""
    return sorted(p.name for p in parquet_dir.glob("*") if p.is_dir())


def _read_frame_for_elr(elr_mil: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return **X**, **Y** DataFrames for one ELR_MIL partition.

    * `Y` to one boolean column per incident type.
    * `X` to all other predictors, plus a categorical `elr_mil` indicator.
    """
    df = build_modelling_frame(elr_mil=elr_mil)
    df = df.drop(columns=["hour", "day", "month", "year", "run_hour"],
                 errors="ignore")
    incident_cols = [c for c in df.columns if c.startswith(INCIDENT_PREFIX)]
    if not incident_cols:
        raise ValueError(f"No INCIDENT_* columns found in partition {elr_mil!r}")

    y = df[incident_cols].astype(bool).astype(int)
    X = df.drop(columns=incident_cols).copy()
    return X, y


def _concatenate_partitions(elrs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all partitions into a single (X, Y).  May use lots of RAM."""
    X_parts, Y_parts = [], []
    for elr in elrs:
        X_i, y_i = _read_frame_for_elr(elr)
        X_parts.append(X_i)
        Y_parts.append(y_i)
    return pd.concat(X_parts, ignore_index=True), pd.concat(Y_parts, ignore_index=True)


def _preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical/object columns; leave numerics untouched."""
    cat_cols = [c for c in X.columns
                if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    return pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ──────────────────────────────────────────────────────────────────────────────
# Core training routine
# ──────────────────────────────────────────────────────────────────────────────

def train(full_X: pd.DataFrame, full_Y: pd.DataFrame,
          seed: int = RANDOM_SEED,
          model_path: Path = MODEL_OUT):
    """Train a `MultiOutputClassifier` of XGBClassifiers and save to *model_path*."""

    X_train, X_val, Y_train, Y_val = train_test_split(
        full_X, full_Y, test_size=0.2, random_state=seed, stratify=None
    )

    base = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="gpu_hist",   # change to "gpu_hist" if a GPU is available
        enable_categorical=False,
        n_jobs=-1,
        random_state=seed,
    )

    clf = MultiOutputClassifier(base, n_jobs=-1)
    log.info("Fitting XGB multi-label model on %s rows × %s labels",
             len(full_X), full_Y.shape[1])
    clf.fit(X_train, Y_train)

    # ── quick per‑label validation ───────────────────────────────────────────
    aucs = {}
    for idx, col in enumerate(full_Y.columns):
        y_true = Y_val.iloc[:, idx]
        y_hat = clf.estimators_[idx].predict_proba(X_val)[:, 1]
        aucs[col] = roc_auc_score(y_true, y_hat)
    avg_auc = sum(aucs.values()) / len(aucs)
    log.info("Mean ROC-AUC across labels: %.3f", avg_auc)

    for col, auc in sorted(aucs.items(), key=lambda kv: kv[0]):
        log.debug("%-30s  AUC: %.3f", col, auc)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    log.info("Saved model → %s", model_path)
    return clf


def train_incremental(elrs: List[str] | None = None,
                      params: dict | None = None,
                      num_boost_round: int = 200,
                      out_json: Path = Path("xgb_incidents_stream.json")):
    """Train separate boosters per label, streaming one partition at a time.

    *Useful when the full table will not fit in RAM.*  A model JSON is written
    for *each* incident label:  `out_json.parent / f"{col}.json"`.
    """
    import xgboost as xgb

    params = params or {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": RANDOM_SEED,
    }

    elrs = elrs or _available_elrs()

    _, sample_y = _read_frame_for_elr(elrs[0])
    label_names = list(sample_y.columns)
    boosters = {lbl: None for lbl in label_names}

    for elr in elrs:
        X_chunk, Y_chunk = _read_frame_for_elr(elr)
        X_chunk = _preprocess(X_chunk)
        for lbl in label_names:
            dtrain = xgb.DMatrix(X_chunk, label=Y_chunk[lbl])
            boosters[lbl] = xgb_train(
                params, dtrain,
                num_boost_round=num_boost_round,
                xgb_model=boosters[lbl],
                verbose_eval=False,
            )
        del X_chunk, Y_chunk, dtrain
        gc.collect()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    for lbl, bst in boosters.items():
        path = out_json.parent / f"{lbl}.json"
        bst.save_model(path)
        log.info("Saved streamed booster → %s", path)

    return boosters
