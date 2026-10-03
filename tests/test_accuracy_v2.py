"""Accuracy experiment boundaries and the authorized course model families."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid

from src.data.common import load_config
from src.models import accuracy_v2 as experiment
from src.models.course_models import FAMILIES, course_grid, json_parameters, make_course_pipeline
from src.models.train import make_folds, save_pipeline


def small_data():
    values, labels = make_classification(n_samples=120, n_features=3, n_informative=2, n_redundant=0, random_state=7)
    X = pd.DataFrame(values, columns=["tempo", "loudness", "duration"])
    X["artist_score"] = (np.arange(120) % 2).astype(int)
    y = pd.Series(labels)
    metadata = pd.DataFrame({"track_id": [f"t{i}" for i in range(120)], "artist_key": [f"a{i//2}" for i in range(120)]})
    config = load_config("configs/accuracy_v2.yaml")
    config["cv"].update(outer_folds=3, inner_folds=2)
    return X, y, metadata, config


def test_threshold_ties_limits_and_validation():
    cutoff, curve = experiment.choose_threshold([0, 0, 1, 1], [.1, .2, .3, .4])
    assert cutoff == .3
    assert curve.accuracy.max() == 1
    cutoff, _ = experiment.choose_threshold([0, 1], [.5, .5])
    assert cutoff == .5
    with pytest.raises(ValueError, match="finite"):
        experiment.choose_threshold([0, 1], [.1, np.nan])


@pytest.mark.parametrize("name", list(FAMILIES))
def test_all_course_models_round_trip_and_record_full_configuration(name, tmp_path):
    X, y, metadata, config = small_data()
    pipeline = make_course_pipeline(name, config)
    first = next(iter(ParameterGrid(course_grid(name, config))))
    pipeline.set_params(**first)
    # Keep unit ensemble tests small without changing production settings.
    if "model__n_estimators" in pipeline.get_params():
        pipeline.set_params(model__n_estimators=10)
    pipeline.fit(X, y)
    evidence = save_pipeline(pipeline, X, tmp_path / f"{name}.joblib")
    assert evidence["round_trip_verified"]
    json.dumps(json_parameters(pipeline), allow_nan=False)
    assert course_grid(name, config) == course_grid(name, config)
    assert len(list(ParameterGrid(course_grid(name, config)))) <= config["models"][name].get("max_candidates", 100)
    if name == "polynomial_logistic":
        assert pipeline.named_steps["polynomial"].n_output_features_ == 14
    with pytest.raises(ValueError, match="scope"):
        make_course_pipeline("neural_network", config)


def test_nested_threshold_and_parameter_selection_only_see_outer_training(monkeypatch, tmp_path):
    X, y, metadata, config = small_data()
    config["models"]["logistic_regression"]["grid"] = {"model__C": [.1, 1.]}
    folds = make_folds(y, metadata, config["protocol"], count=3, seed=42)
    original = experiment.tune_stage
    observed = []

    def recording(pipeline, grid, fitting_X, fitting_y, fitting_metadata, cfg, seed):
        observed.append(set(fitting_X.index))
        for fitting, validation in make_folds(fitting_y, fitting_metadata, cfg["protocol"], count=2, seed=seed):
            assert set(fitting_metadata.iloc[fitting].artist_key).isdisjoint(fitting_metadata.iloc[validation].artist_key)
        return original(pipeline, grid, fitting_X, fitting_y, fitting_metadata, cfg, seed)

    monkeypatch.setattr(experiment, "tune_stage", recording)
    # train_family requires artifact paths beneath its recorded repository root.
    monkeypatch.setattr(experiment, "repo_path", lambda name: tmp_path / name)
    models, reports = tmp_path / "models", tmp_path / "reports"
    models.mkdir(); reports.mkdir()
    summary, artifacts = experiment.train_family("logistic_regression", X, y, metadata, folds, config, models, reports)
    for seen, (fitting, validation) in zip(observed, folds):
        assert seen == set(X.iloc[fitting].index)
        assert seen.isdisjoint(X.iloc[validation].index)
    assert observed[-1] == set(X.index)
    assert len(artifacts) == 4
    assert 0 <= summary["cv_accuracy_mean"] <= 1


def test_entry_point_never_requests_old_test_data(monkeypatch, tmp_path):
    _, _, _, config = small_data()
    monkeypatch.setattr(experiment, "load_config", lambda _: config)
    monkeypatch.setattr(experiment, "repo_path", lambda name: tmp_path / name)
    monkeypatch.setattr(experiment, "sha256", lambda _: "hash")
    monkeypatch.setattr(experiment, "source_names", lambda _: [])
    monkeypatch.setattr(experiment, "snapshot_sources", lambda *args: {})
    monkeypatch.setattr(experiment, "verify_handoff", lambda: {"integrity_passed": True, "input_sha256": "hash"})
    requested = []

    def loader(protocol, partition, **kwargs):
        requested.append((partition, kwargs))
        raise RuntimeError("Training loader boundary")

    monkeypatch.setattr(experiment, "load_partition", loader)
    with pytest.raises(RuntimeError, match="boundary"):
        experiment.run("guard")
    assert requested == [("train", {})]
