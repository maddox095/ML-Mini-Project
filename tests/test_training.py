"""Checks for leakage boundaries, feature ablation and saved model integrity."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.data.common import load_config
from src.features.build_features import make_pipeline, make_preprocessor
from src.models.evaluate import binary_metrics, prediction_scores, score_metrics, scoring_for
from src.models.compare import compatible_manifests, require_matching_folds
from src.models.registry import MODELS, TREE_MODELS, make_estimator, search_space
from sklearn.model_selection import ParameterGrid
from src.models import train


@pytest.fixture
def training_data():
    values, labels = make_classification(n_samples=120, n_features=3, n_informative=2,
        n_redundant=0, random_state=42)
    X = pd.DataFrame(values, columns=["tempo", "loudness", "duration"])
    X["artist_score"] = (np.arange(len(X)) % 3 == 0).astype(int)
    X["hit"] = labels
    y = pd.Series(labels)
    metadata = pd.DataFrame({"track_id": [f"track{i}" for i in range(len(X))],
                             "artist_key": [f"artist{i // 4}" for i in range(len(X))]})
    return X, y, metadata


def small_config():
    config = load_config("configs/models.yaml")
    config["cv"].update(outer_folds=3, inner_folds=2)
    config["logistic_regression"]["C_grid"] = [0.1, 1.0]
    return config


@pytest.mark.parametrize("features", [
    ["tempo", "loudness", "duration"], ["artist_score"], ["tempo", "artist_score"],
])
def test_ablation_pipeline_uses_only_requested_whitelisted_inputs(training_data, features):
    X, _, _ = training_data
    transformed = make_preprocessor(features=features).fit_transform(X)
    assert transformed.shape == (len(X), len(features))
    if "artist_score" in features:
        np.testing.assert_array_equal(transformed[:, -1], X.artist_score)
    with pytest.raises(ValueError, match="whitelist"):
        make_preprocessor(features=["tempo", "hit"])


@pytest.mark.parametrize("model", ["logistic_regression", "svm_linear"])
def test_nested_cv_keeps_validation_out_of_tuning_and_preprocessing(training_data, monkeypatch, model):
    X, y, metadata = training_data
    config = small_config()
    folds = train.make_folds(y, metadata, "artist_disjoint_partition", count=3, seed=42)
    real_fit = train.fit_candidate
    observed_indices = []

    def recording_fit(pipeline, grid, fitting_X, fitting_y, inner_folds, configuration):
        for fitting, validation in inner_folds:
            training_artists = set(metadata.loc[fitting_X.iloc[fitting].index, "artist_key"])
            validation_artists = set(metadata.loc[fitting_X.iloc[validation].index, "artist_key"])
            assert training_artists.isdisjoint(validation_artists)
        fitted, searches, seconds = real_fit(pipeline, grid, fitting_X, fitting_y, inner_folds, configuration)
        scaler = fitted.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, fitting_X[["tempo", "loudness", "duration"]].mean())
        observed_indices.append(set(fitting_X.index))
        return fitted, searches, seconds

    monkeypatch.setattr(train, "fit_candidate", recording_fit)
    result = train.train_experiment(model,
        ["tempo", "loudness", "duration", "artist_score"], X, y, metadata,
        "artist_disjoint_partition", folds, config)
    for seen, (fitting, validation) in zip(observed_indices, folds):
        assert seen == set(X.iloc[fitting].index)
        assert seen.isdisjoint(X.iloc[validation].index)
    assert observed_indices[-1] == set(X.index)
    assert result["oof"].track_id.is_unique
    score_kind = "probability" if model == "logistic_regression" else "decision_score"
    assert result["oof"][score_kind].notna().all()
    if model == "svm_linear":
        assert "probability" not in result["oof"]
        assert "cv_log_loss_mean" not in result["summary"]
    assert len(result["oof"]) == len(X)


@pytest.mark.parametrize("model", list(MODELS))
def test_saved_pipeline_round_trip(training_data, tmp_path, model):
    X, y, _ = training_data
    pipeline = make_pipeline(make_estimator(model, small_config()), scale=model not in TREE_MODELS).fit(X, y)
    path = tmp_path / "pipeline.joblib"
    evidence = train.save_pipeline(pipeline, X, path)
    assert evidence["round_trip_verified"]
    score_kind = "decision_score" if model.startswith("svm_") else "probability"
    assert evidence[f"max_{score_kind}_difference"] == 0
    np.testing.assert_allclose(prediction_scores(joblib.load(path), X)[0], prediction_scores(pipeline, X)[0])


def test_search_caps_are_deterministic_and_network_matches_architecture(training_data):
    config = small_config()
    for model in MODELS:
        grid = search_space(model, config)
        assert grid == search_space(model, config)
        candidates = list(ParameterGrid(grid))
        assert len(candidates) <= config[model].get("search_max_candidates", len(candidates))
        assert all(key.startswith("model__") for key in candidates[0])
    X, y, _ = training_data
    pipeline = make_pipeline(make_estimator("neural_network", config)).fit(X, y)
    network = pipeline.named_steps["model"]
    assert [weight.shape for weight in network.coefs_] == [(4, 6), (6, 1)]
    assert network.activation == network.out_activation_ == "logistic"
    tree = make_pipeline(make_estimator("decision_tree", config), scale=False).fit(X, y)
    assert "scaler" not in tree.named_steps["preprocessor"].named_transformers_["numeric"].named_steps


def test_metrics_handle_dummy_predictions_and_reject_invalid_probabilities():
    result = binary_metrics([0, 1, 0, 1], [0.5] * 4, predictions=[0] * 4)
    assert result["accuracy"] == 0.5
    assert result["roc_auc"] == 0.5
    assert result["precision"] == result["recall"] == 0
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (2, 0, 2, 0)
    with pytest.raises(ValueError, match="finite"):
        binary_metrics([0, 1], [0.5, np.nan])


@pytest.mark.parametrize("model", ["logistic_regression", "svm_linear"])
def test_training_entry_point_only_requests_training_partitions(monkeypatch, tmp_path, model):
    # Stop immediately after the loader boundary; no final test access is possible.
    config = small_config()
    monkeypatch.setattr(train, "load_config", lambda _: config)
    monkeypatch.setattr(train, "repo_path", lambda path: tmp_path / path)
    monkeypatch.setattr(train, "sha256", lambda _: "test_hash")
    monkeypatch.setattr(train, "snapshot_sources", lambda names, root: {name: "test_hash" for name in names})
    monkeypatch.setattr(train, "verify_handoff", lambda: {
        "integrity_passed": True, "input_sha256": "test_hash", "dataset_version": "test",
        "raw_inputs_verified": False, "limitations": []})
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    (processed / "user2_split_manifest.json").write_text('{"splits_sha256": "test_hash"}')
    requested = []

    def loader(protocol, partition, **kwargs):
        requested.append((protocol, partition, kwargs))
        raise RuntimeError("training loader boundary reached")

    monkeypatch.setattr(train, "load_partition", loader)
    with pytest.raises(RuntimeError, match="boundary reached"):
        train.run(run_id="loader_test", model=model)
    assert requested == [("stratified_partition", "train", {})]


def test_margin_metrics_use_raw_scores_without_probability_metrics(training_data):
    X, y, _ = training_data
    pipeline = make_pipeline(make_estimator("svm_linear", small_config())).fit(X, y)
    values, kind = prediction_scores(pipeline, X)
    assert kind == "decision_score"
    assert np.any(values < 0)
    metrics = score_metrics(y, values, predictions=pipeline.predict(X), score_kind=kind)
    assert "log_loss" not in metrics and "brier_score" not in metrics
    assert "neg_log_loss" not in scoring_for(pipeline)
    assert "neg_log_loss" in scoring_for(make_pipeline(make_estimator("logistic_regression", small_config())))


def test_source_snapshots_preserve_uncommitted_run_code(monkeypatch, tmp_path):
    source = tmp_path / "repository/code.py"
    source.parent.mkdir()
    source.write_text("model = 'svm_linear'\n", encoding="utf-8")
    monkeypatch.setattr(train, "repo_path", lambda name: source.parent / name)
    recorded = train.snapshot_sources(["code.py"], tmp_path / "run")
    snapshot = tmp_path / "run/source_snapshot/code.py"
    assert snapshot.read_bytes() == source.read_bytes()
    source.write_text("model = 'changed'\n", encoding="utf-8")
    assert snapshot.read_text() == "model = 'svm_linear'\n"
    assert train.sha256(snapshot) == recorded["code.py"]


def test_comparison_rejects_different_data_or_validation():
    config = small_config()
    manifest = {"source_sha256": "data", "splits_sha256": "splits",
                "dataset_version": "v1", "test_sets_evaluated": False, "config": config}
    assert compatible_manifests(manifest, manifest)
    assert not compatible_manifests(manifest, {**manifest, "source_sha256": "changed"})
    changed_config = {**config, "cv": {**config["cv"], "outer_folds": 4}}
    assert not compatible_manifests(manifest, {**manifest, "config": changed_config})
    assert not compatible_manifests(manifest, {**manifest, "test_sets_evaluated": True})


def test_comparison_rejects_mismatched_validation_assignments():
    left = pd.DataFrame({"model": ["logistic_regression"] * 2, "feature_set": ["audio_only"] * 2,
        "protocol": ["artist_disjoint_partition"] * 2, "track_id": ["a", "b"],
        "artist_key": ["artist_a", "artist_b"], "hit": [0, 1], "fold": [1, 2]})
    right = left.assign(model="svm_linear")
    require_matching_folds(left, "logistic_regression", right, "svm_linear")
    right.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="identical"):
        require_matching_folds(left, "logistic_regression", right, "svm_linear")
