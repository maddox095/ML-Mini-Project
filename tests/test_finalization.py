"""Protect the final-test boundary and independently replay saved settings."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build_features import make_pipeline
from src.models import finalize, replay
from src.models.train import write_json
from src.data.common import load_config, sha256


def test_candidate_uses_highest_cv_accuracy_not_simplicity_or_test_scores():
    config = load_config("configs/models.yaml")
    config.pop("deployment_model")
    rows = []
    for model, auc in [("logistic_regression", .875), ("neural_network", .880)]:
        for feature_set, penalty in [("audio_only", .1), ("audio_artist_score", 0)]:
            rows.append({"model": model, "protocol": "artist_disjoint_partition", "feature_set": feature_set,
                "cv_roc_auc_mean": auc - penalty, "run_id": model, "artifact": model,
                "cv_accuracy_mean": (.80 if model == "logistic_regression" else .84) - penalty,
                "test_accuracy": 1 if model == "logistic_regression" else 0})
    choice = finalize.choose_candidate(pd.DataFrame(rows), config)
    assert choice["model"] == "neural_network"
    assert choice["feature_set"] == "audio_artist_score"
    assert choice["best_cv_roc_auc_mean"] == .880


def test_deployment_family_selects_best_validation_feature_set():
    config = load_config("configs/models.yaml")
    rows = pd.DataFrame([
        {"model": "random_forest", "feature_set": "audio_only", "cv_accuracy_mean": .65},
        {"model": "random_forest", "feature_set": "audio_artist_score", "cv_accuracy_mean": .82},
        {"model": "decision_tree", "feature_set": "audio_artist_score", "cv_accuracy_mean": .81},
    ]).assign(protocol="artist_disjoint_partition", cv_roc_auc_mean=.88, run_id="test", artifact="test")
    selected = finalize.choose_candidate(rows, config)
    assert selected["model"] == "random_forest"
    assert selected["feature_set"] == "audio_artist_score"
    assert selected["cv_accuracy_mean"] == .82


def test_no_test_access_without_exact_frozen_file(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(finalize, "load_partition", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="frozen"):
        finalize.evaluate_frozen({"chosen": {}}, tmp_path)
    write_json(tmp_path / "selection.json", {"chosen": {"model": "changed"}})
    with pytest.raises(ValueError, match="frozen"):
        finalize.evaluate_frozen({"chosen": {}}, tmp_path)
    assert not calls


def test_frozen_evaluation_never_fits_or_changes_selection(tmp_path, monkeypatch):
    X = pd.DataFrame({"tempo": [90, 110, 130, 150], "loudness": [-20, -10, -8, -4],
                      "duration": [120, 150, 180, 210], "artist_score": [0, 0, 1, 1]})
    y = pd.Series([0, 0, 1, 1])
    metadata = pd.DataFrame({"track_id": list("abcd"), "artist_key": list("abcd"),
                             "artist_name": list("abcd"), "title": list("abcd")})
    path = tmp_path / "pipeline.joblib"
    pipeline = make_pipeline(LogisticRegression()).fit(X, y)
    joblib.dump(pipeline, path)
    identity = {"protocol": "artist_disjoint_partition", "model": "logistic_regression",
                "feature_set": "audio_artist_score", "run_id": "lr"}
    selection = {"chosen": identity, "protocols": [identity["protocol"]],
        "tuple_parameter": (6,),
        "artifacts": [{**identity, "path": str(path), "sha256": sha256(path)}]}
    write_json(tmp_path / "selection.json", selection)
    calls = []

    def loader(protocol, partition, **kwargs):
        calls.append((protocol, partition, kwargs))
        assert json.loads((tmp_path / "selection.json").read_text()) == json.loads(json.dumps(selection))
        return X, y, metadata

    monkeypatch.setattr(finalize, "load_partition", loader)
    monkeypatch.setattr(LogisticRegression, "fit", lambda *args, **kwargs: pytest.fail("Test code must never fit"))
    result, predictions = finalize.evaluate_frozen(selection, tmp_path)
    assert result.selected_before_test.tolist() == [True]
    assert len(predictions) == len(X)
    assert calls == [(identity["protocol"], "test", {"allow_test": True})]
    assert json.loads((tmp_path / "selection.json").read_text()) == json.loads(json.dumps(selection))


@pytest.mark.parametrize("model,parameters", [
    ("logistic_regression", {"model__C": .1}), ("svm_linear", {"model__C": .1}),
    ("svm_rbf", {"model__C": 1, "model__gamma": "scale"}),
    ("svm_polynomial", {"model__C": 1, "model__degree": 2, "model__coef0": 1}),
    ("decision_tree", {"model__max_depth": 2}),
    ("random_forest", {"model__max_depth": 4}),
    ("neural_network", {"model__alpha": .01}),
])
def test_replay_reconstructs_folds_without_requesting_test(tmp_path, monkeypatch, model, parameters):
    n = 60
    X = pd.DataFrame({"tempo": [90, 110] * (n // 2), "loudness": [-10, -5] * (n // 2),
                      "duration": [120, 200] * (n // 2), "artist_score": [0, 1] * (n // 2)})
    y = pd.Series([0, 1] * (n // 2))
    metadata = pd.DataFrame({"track_id": [str(i) for i in range(n)], "artist_key": [str(i // 2) for i in range(n)]})
    config = load_config("configs/models.yaml")
    config["cv"].update(outer_folds=3, inner_folds=2)
    setting = {"config": config, "protocol": "artist_disjoint_partition", "stage": "outer_fold_1",
        "model": model, "features": config["feature_sets"]["audio_artist_score"],
        "parameters": parameters, "source_sha256": "test"}
    path = tmp_path / "setting.json"
    write_json(path, setting)
    calls = []

    def loader(protocol, partition):
        calls.append(partition)
        return X, y, metadata

    monkeypatch.setattr(replay, "load_partition", loader)
    monkeypatch.setattr(replay, "sha256", lambda _: "test")
    result = replay.run(path, tmp_path / "output", inner_fold=1)
    assert calls == ["train"]
    assert result["training_rows"] == 20
    for key, value in parameters.items():
        assert result["estimator_parameters"][key.removeprefix("model__")] == value
    assert result["round_trip_verified"]
    assert result["validation_metrics"]["accuracy"] == 1
