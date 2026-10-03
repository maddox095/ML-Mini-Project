"""Replay a saved course-scoped hyperparameter fit or evaluate saved CV models.

These commands load only the original training partition, never the v1 tests.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.common import repo_path, sha256
from src.models.course_models import json_parameters, make_course_pipeline
from src.models.data import load_partition
from src.models.evaluate import prediction_scores, score_metrics
from src.models.artifacts import save_pipeline, write_json
from src.models.validation import make_folds, replay_partition


def replay(setting_path, output, inner_fold=None):
    setting = json.loads(Path(setting_path).read_text(encoding="utf-8"))
    if (sha256(repo_path("data/interim/model_table.parquet")) != setting["source_sha256"]
            or sha256(repo_path("data/processed/user2_splits.csv")) != setting["splits_sha256"]):
        raise ValueError("Saved experiment inputs changed")
    config = setting["config"]
    X, y, metadata = load_partition(config["protocol"], "train")
    X, y, validation_X, validation_y = replay_partition(
        X, y, metadata, protocol=config["protocol"], stage=setting["stage"],
        config=config, inner_fold=inner_fold)
    pipeline = make_course_pipeline(setting["model"], config).set_params(**setting["parameters"])
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new replay output folder")
    output.mkdir(parents=True)
    with threadpool_limits(limits=1):
        pipeline.fit(X, y)
    result = {"setting": setting, "training_rows": len(y), "inner_fold": inner_fold,
        "pipeline_configuration": json_parameters(pipeline),
        **save_pipeline(pipeline, X.iloc[:32], output / "pipeline.joblib")}
    if validation_X is not None:
        values, kind = prediction_scores(pipeline, validation_X)
        result["validation_metrics"] = score_metrics(validation_y, values,
            predictions=pipeline.predict(validation_X), score_kind=kind)
    write_json(output / "metadata.json", result)
    print(f"Saved fixed-parameter course model and results: {output}", flush=True)
    return result


def evaluate_saved_cv(run_id, model):
    root = repo_path(f"reports/runs/{run_id}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (sha256(repo_path("data/interim/model_table.parquet")) != manifest["source_sha256"]
            or sha256(repo_path("data/processed/user2_splits.csv")) != manifest["splits_sha256"]):
        raise ValueError("Saved experiment inputs changed")
    config = manifest["config"]
    X, y, metadata = load_partition(config["protocol"], "train")
    folds = make_folds(y, metadata, config["protocol"], count=config["cv"]["outer_folds"], seed=config["seed"])
    rows = []
    for entry in manifest["artifacts"]:
        if entry["model"] != model or not entry["stage"].startswith("outer_fold_"):
            continue
        if sha256(repo_path(entry["path"])) != entry["sha256"]:
            raise ValueError("Saved outer-fold model changed")
        number = int(entry["stage"].rsplit("_", 1)[-1])
        _, validation = folds[number-1]
        estimator = joblib.load(repo_path(entry["path"]))
        values, kind = prediction_scores(estimator, X.iloc[validation])
        rows.append({"model": model, "fold": number, "threshold": entry["threshold"],
            **score_metrics(y.iloc[validation], values, predictions=estimator.predict(X.iloc[validation]), score_kind=kind)})
    frame = pd.DataFrame(rows)
    if len(frame) != len(folds):
        raise ValueError("A saved outer-fold checkpoint is missing")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting")
    parser.add_argument("--output", default="course_replay")
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--evaluate-run")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.evaluate_run:
        print(evaluate_saved_cv(args.evaluate_run, args.model).to_string(index=False))
    elif args.setting:
        replay(args.setting, args.output, args.inner_fold)
    else:
        parser.error("Choose a saved setting or an evaluation run/model")
