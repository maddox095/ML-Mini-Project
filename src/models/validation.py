"""Shared training-only folds, parameter search and replay partition selection."""

import json
import re
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from src.models.data import PROTOCOLS
from src.models.evaluate import scoring_for


def make_folds(y: pd.Series, metadata: pd.DataFrame, protocol: str,
               *, count: int, seed: int) -> list:
    """Use positional indices and verify coverage, class balance and groups."""
    if protocol not in PROTOCOLS or count < 2 or len(y) != len(metadata):
        raise ValueError("Invalid cross-validation protocol or row count")
    grouped = protocol == "artist_disjoint_partition"
    splitter = (StratifiedGroupKFold if grouped else StratifiedKFold)(
        n_splits=count, shuffle=True, random_state=seed)
    groups = metadata.artist_key.to_numpy() if grouped else None
    folds = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    seen = np.zeros(len(y), dtype=int)
    for training, validation in folds:
        if np.intersect1d(training, validation).size:
            raise ValueError("A cross-validation row appears in both partitions")
        if any(set(y.iloc[indices].unique()) != {0, 1} for indices in (training, validation)):
            raise ValueError("Every cross-validation partition must contain both classes")
        if grouped and set(groups[training]) & set(groups[validation]):
            raise ValueError("Artist leakage in cross-validation")
        seen[validation] += 1
    if not np.all(seen == 1):
        raise ValueError("Cross-validation must score each training row exactly once")
    return folds


def fit_candidate(pipeline, grid: dict | list, X, y, folds, config: dict):
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if grid:
            search = GridSearchCV(
                pipeline, grid, scoring=scoring_for(pipeline), refit=config["cv"]["selection_metric"],
                cv=folds, n_jobs=config["cv"]["n_jobs"], error_score="raise")
            search.fit(X, y)
            fitted = search.best_estimator_
            results = pd.DataFrame(search.cv_results_)
            results["params"] = results.params.map(lambda value: json.dumps(value, sort_keys=True))
        else:
            fitted = clone(pipeline).fit(X, y)
            results = pd.DataFrame()
    return fitted, results, perf_counter() - started


def replay_partition(X, y, metadata, *, protocol, stage, config, inner_fold=None):
    """Reconstruct a recorded fit's rows without loading or fitting any model."""
    validation_X = validation_y = None
    seed = config["seed"]
    if stage != "full_training_tuning":
        match = re.fullmatch(r"outer_fold_([1-9][0-9]*)", stage)
        if match is None:
            raise ValueError("Unknown replay stage")
        number = int(match.group(1))
        if number > config["cv"]["outer_folds"]:
            raise ValueError("Outer fold out of range")
        folds = make_folds(y, metadata, protocol, count=config["cv"]["outer_folds"], seed=seed)
        fitting, validation = folds[number - 1]
        validation_X, validation_y = X.iloc[validation], y.iloc[validation]
        X, y, metadata = X.iloc[fitting], y.iloc[fitting], metadata.iloc[fitting]
        seed += number
    if inner_fold is not None:
        if inner_fold < 1 or inner_fold > config["cv"]["inner_folds"]:
            raise ValueError("Inner fold out of range")
        folds = make_folds(y, metadata, protocol, count=config["cv"]["inner_folds"], seed=seed)
        fitting, validation = folds[inner_fold - 1]
        validation_X, validation_y = X.iloc[validation], y.iloc[validation]
        X, y = X.iloc[fitting], y.iloc[fitting]
    return X, y, validation_X, validation_y
