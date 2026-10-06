"""Publish the already evaluated v1 forest without fitting or rewriting history."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.common import sha256
from src.models.artifacts import write_json

VERSION = "random_forest_v1_deployment"
IDENTITY = {"model": "random_forest", "protocol": "artist_disjoint_partition",
            "feature_set": "audio_artist_score"}


def main():
    frozen = json.loads((ROOT / "reports/runs/final_suite_v1/selection.json").read_text())
    artifact = next(item for item in frozen["artifacts"]
                    if all(item[key] == value for key, value in IDENTITY.items()))
    source = ROOT / artifact["path"]
    if sha256(source) != artifact["sha256"]:
        raise ValueError("Original forest does not match its frozen checksum")
    metrics = pd.read_csv(ROOT / "reports/runs/final_suite_v1/test_metrics.csv")
    primary = metrics.loc[(metrics.protocol == IDENTITY["protocol"])
                          & (metrics.feature_set == IDENTITY["feature_set"])
                          & (metrics.model != "dummy")]
    forest = primary.loc[primary.model == IDENTITY["model"]].iloc[0]
    if not np.isclose(forest.accuracy, primary.accuracy.max(), atol=1e-12):
        raise ValueError("Forest is not the highest-accuracy evaluated primary model")
    pipeline = joblib.load(source)
    if not isinstance(pipeline.named_steps["model"], RandomForestClassifier):
        raise ValueError("Frozen artifact is not a random forest")
    active = ROOT / "models/best_pipeline.joblib"
    metadata_path = ROOT / "models/metadata.json"
    previous = json.loads(metadata_path.read_text())
    if previous["final_run_id"] != VERSION:
        if sha256(active) != previous["sha256"]:
            raise ValueError("Previous deployment checksum changed")
        archive = ROOT / "models/deployments" / previous["final_run_id"]
        archive.mkdir(parents=True, exist_ok=True)
        for path in (active, metadata_path):
            target = archive / path.name
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise FileExistsError(f"Existing historical deployment differs: {target}")
            if not target.exists():
                shutil.copyfile(path, target)

    cv = pd.read_csv(ROOT / "reports/model_comparison.csv")
    cv_forest = cv.loc[(cv.model == IDENTITY["model"])
                      & (cv.protocol == IDENTITY["protocol"])
                      & (cv.feature_set == IDENTITY["feature_set"])].iloc[0]
    choice = {**IDENTITY, "run_id": artifact["run_id"], "artifact": artifact["path"],
              "cv_accuracy_mean": float(cv_forest.cv_accuracy_mean),
              "cv_roc_auc_mean": float(cv_forest.cv_roc_auc_mean),
              "selected_before_test": False,
              "policy": "User-authorized deployment of the saved random forest after reviewing the completed v1 comparison; highest observed primary held-out accuracy. This is a deployment change, not a new independent evaluation."}
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")
    metadata = {"selection": choice, "features": artifact["features"],
                "threshold": artifact["threshold"], "positive_class_rule": "probability > 0.5; exact ties predict non-hit",
                "units": previous["units"],
                "probability_interpretation": "Balanced research-sample probability; not independently calibrated and not a natural-release commercial-success probability.",
                "estimator_parameters": artifact["estimator_parameters"],
                "source_sha256": frozen["source_sha256"],
                "splits_sha256": frozen["splits_sha256"],
                "final_run_id": VERSION, "evaluation_run_id": "final_suite_v1",
                "training_rows": artifact["training_rows"],
                "test_rows": int(forest.test_rows),
                "test_metrics": {key: float(forest[key]) for key in keys},
                "confusion_matrix": {key: int(forest[key]) for key in ("tn", "fp", "fn", "tp")},
                "feature_importances": dict(zip(artifact["features"],
                    map(float, pipeline.named_steps["model"].feature_importances_))),
                "sha256": artifact["sha256"], "score_kind": "probability",
                "round_trip_verified": True, "max_probability_difference": 0.0,
                "deployment_updated_at_utc": datetime.now(UTC).isoformat()}
    sample = pd.DataFrame([[120, -8, 210, 1], [90, -16, 180, 0]], columns=metadata["features"])
    original_probabilities = pipeline.predict_proba(sample)
    shutil.copyfile(source, active)
    restored = joblib.load(active)
    np.testing.assert_allclose(restored.predict_proba(sample), original_probabilities, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(restored.predict(sample), pipeline.predict(sample))
    write_json(metadata_path, metadata)
    deployment = ROOT / "models/deployments" / VERSION
    deployment.mkdir(parents=True, exist_ok=True)
    write_json(deployment / "metadata.json", metadata)
    write_json(deployment / "promotion.json", {
        "version": VERSION, "source_artifact": artifact,
        "source_selection": "reports/runs/final_suite_v1/selection.json",
        "source_metrics": "reports/runs/final_suite_v1/test_metrics.csv",
        "historical_selection_unchanged": True, "fitting_performed": False,
        "fresh_test_performed": False, "selected_before_test": False,
        "active_sha256": sha256(active), "metrics": metadata["test_metrics"],
        "confusion_matrix": metadata["confusion_matrix"]})
    print(json.dumps({"model": choice["model"], "version": VERSION,
                      "metrics": metadata["test_metrics"],
                      "confusion_matrix": metadata["confusion_matrix"]}, indent=2))


if __name__ == "__main__":
    main()
