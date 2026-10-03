"""Accuracy-focused, course-scoped development using training-only nested CV.

Run: python -m src.models.accuracy_v2 --run-id course_accuracy_v2
No old test partition is loaded. Each outer validation fold remains separate
from hyperparameter selection, threshold selection and preprocessing fitting.
"""

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
import platform
import re
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import FixedThresholdClassifier
from threadpoolctl import threadpool_limits

from src.data.common import load_config, repo_path, sha256
from src.data.verify_handoff import verify_handoff
from src.models.course_models import FAMILIES, course_grid, json_parameters, make_course_pipeline
from src.models.data import load_partition
from src.models.evaluate import prediction_scores, score_metrics
from src.models.artifacts import save_pipeline, snapshot_sources, source_files, write_json
from src.models.validation import fit_candidate, make_folds


def choose_threshold(y, probabilities):
    """Fit accuracy cutoff on fitting-data OOF scores, with a declared tie rule."""
    labels, scores = np.asarray(y), np.asarray(probabilities, dtype=float)
    if (labels.shape != scores.shape or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any() or set(np.unique(labels)) != {0, 1}):
        raise ValueError("Aligned binary labels and finite probabilities required")
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    thresholds = np.unique(np.r_[scores, 0.5, 0.0, 1.0, np.nextafter(1.0, np.inf)])
    offsets = np.searchsorted(sorted_scores, thresholds, side="left")
    positives = np.r_[0, np.cumsum(sorted_labels)]
    below_positive = positives[offsets]
    tp = positives[-1] - below_positive
    tn = offsets - below_positive
    frame = pd.DataFrame({"threshold": thresholds, "accuracy": (tp + tn)/len(labels),
        "tn": tn, "fp": len(labels)-positives[-1]-tn, "fn": below_positive, "tp": tp})
    frame["distance_to_0.5"] = abs(frame.threshold - 0.5)
    best = frame.sort_values(["accuracy", "distance_to_0.5", "threshold"], ascending=[False, True, True]).iloc[0]
    return float(best.threshold), frame


def tune_stage(pipeline, grid, X, y, metadata, config, seed):
    folds = make_folds(y, metadata, config["protocol"], count=config["cv"]["inner_folds"], seed=seed)
    fitted, search, seconds = fit_candidate(pipeline, grid, X, y, folds, config)
    oof = np.full(len(y), np.nan)
    for training, validation in folds:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            inner = clone(fitted).fit(X.iloc[training], y.iloc[training])
        values, kind = prediction_scores(inner, X.iloc[validation])
        if kind != "probability":
            raise ValueError("Course threshold selection requires probabilities")
        oof[validation] = values
    threshold, curve = choose_threshold(y, oof)
    wrapper = FixedThresholdClassifier(fitted, threshold=threshold, response_method="predict_proba")
    # Wrapper prediction needs no refit; it delegates to the already fitted pipeline.
    np.testing.assert_array_equal(wrapper.predict(X), (prediction_scores(fitted, X)[0] >= threshold).astype(int))
    keys = list(grid) if isinstance(grid, dict) else list(grid[0])
    parameters = {key: fitted.get_params()[key] for key in keys}
    return wrapper, fitted, search, curve, parameters, seconds


def train_family(name, X, y, metadata, folds, config, model_root, report_root):
    pipeline, grid = make_course_pipeline(name, config), course_grid(name, config)
    rows, searches, curves, artifacts, selected_settings = [], [], [], [], []
    oof = metadata[["track_id", "artist_key"]].reset_index(drop=True).copy()
    oof["hit"] = y.to_numpy()
    for column in ["probability", "fold_threshold"]:
        oof[column] = np.nan
    for column in ["fold", "prediction", "default_prediction"]:
        oof[column] = -1
    started = perf_counter()
    stages = [(f"outer_fold_{number}", training, validation, config["seed"]+number)
              for number, (training, validation) in enumerate(folds, 1)]
    stages.append(("full_training_tuning", np.arange(len(y)), None, config["seed"]))
    for stage, training, validation, seed in stages:
        print(f"Training {name}: {stage} ({len(training):,} fitting rows)", flush=True)
        wrapper, fitted, search, curve, parameters, seconds = tune_stage(
            pipeline, grid, X.iloc[training], y.iloc[training], metadata.iloc[training], config, seed)
        search["params"] = search["params"].map(str)
        searches.append(search.assign(model=name, stage=stage))
        curves.append(curve.assign(model=name, stage=stage))
        artifact = model_root / f"{name}__{stage}.joblib"
        evidence = save_pipeline(wrapper, X.iloc[training[:32]], artifact)
        entry = {"model": name, "stage": stage, "path": artifact.relative_to(repo_path(".")).as_posix(),
            "threshold": wrapper.threshold, "comparison": ">=", "features": config["features"],
            "training_rows": len(training), "tuned_parameters": parameters,
            "pipeline_configuration": json_parameters(fitted), **evidence}
        artifacts.append(entry)
        selected_settings.append({"stage": stage, "threshold": wrapper.threshold, "parameters": parameters})
        if validation is not None:
            values, kind = prediction_scores(wrapper, X.iloc[validation])
            predicted, default = wrapper.predict(X.iloc[validation]), fitted.predict(X.iloc[validation])
            metrics = score_metrics(y.iloc[validation], values, predictions=predicted, score_kind=kind)
            number = int(stage.rsplit("_", 1)[-1])
            rows.append({"model": name, "fold": number, "threshold": wrapper.threshold,
                "selected_parameters": json.dumps(parameters, sort_keys=True), "fit_seconds": seconds,
                "default_accuracy": float(np.mean(default == y.iloc[validation].to_numpy())), **metrics})
            oof.loc[validation, ["probability", "fold_threshold", "fold", "prediction", "default_prediction"]] = np.column_stack(
                [values, np.full(len(validation), wrapper.threshold), np.full(len(validation), number), predicted, default])
        else:
            final = entry
            train_accuracy = float(np.mean(wrapper.predict(X) == y.to_numpy()))
    if oof.probability.isna().any() or oof.fold.lt(1).any():
        raise ValueError("Incomplete outer validation predictions")
    fold_frame = pd.DataFrame(rows)
    summary = {"model": name, "cv_accuracy_mean": float(fold_frame.accuracy.mean()),
        "cv_accuracy_std": float(fold_frame.accuracy.std(ddof=1)),
        "cv_default_accuracy_mean": float(fold_frame.default_accuracy.mean()),
        "cv_roc_auc_mean": float(fold_frame.roc_auc.mean()), "cv_f1_mean": float(fold_frame.f1.mean()),
        "train_accuracy": train_accuracy, "fit_seconds": perf_counter()-started,
        "final_threshold": final["threshold"], "final_parameters": json.dumps(final["tuned_parameters"], sort_keys=True),
        "artifact": final["path"], "reaches_85_percent_cv": bool(fold_frame.accuracy.mean() >= .85)}
    family_root = report_root / name
    family_root.mkdir()
    fold_frame.to_csv(family_root / "fold_metrics.csv", index=False)
    oof.assign(model=name).to_csv(family_root / "outer_predictions.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(family_root / "search_results.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(family_root / "threshold_search.csv", index=False)
    write_json(family_root / "selected_settings.json", {"stages": selected_settings, "artifacts": artifacts})
    return summary, artifacts


def source_names(config_path):
    return source_files(repo_path("."))


def run(run_id="course_accuracy_v2", config_path="configs/accuracy_v2.yaml"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    config = load_config(config_path)
    if (config["cv"]["selection_metric"] != "accuracy" or config["protocol"] != "artist_disjoint_partition"
            or set(config["family_order"]) != set(config["models"]) or not set(config["models"]).issubset(FAMILIES)):
        raise ValueError("Use accuracy, artist-group validation and authorized model families")
    verification = verify_handoff()
    if not verification["integrity_passed"]:
        raise ValueError("Handoff integrity failed")
    report_root, model_root = repo_path(f"reports/runs/{run_id}"), repo_path(f"models/{run_id}")
    if report_root.exists() or model_root.exists():
        raise FileExistsError("Preserve existing runs; use a new ID")
    report_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    split_path = repo_path("data/processed/user2_splits.csv")
    manifest = {"run_id": run_id, "experiment_type": "course_accuracy_v2", "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(), "config": config,
        "source_sha256": verification["input_sha256"], "splits_sha256": sha256(split_path),
        "test_sets_evaluated": False, "existing_test_sets_previously_observed": True,
        "selection_metric": "accuracy", "scope": "regression, decision trees, bagging, boosting",
        "python": platform.python_version(), "device": "cpu",
        "packages": {name: importlib.metadata.version(name) for name in ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]},
        "source_hashes": snapshot_sources(source_names(config_path), report_root), "artifacts": [], "completed_families": []}
    write_json(report_root / "manifest.json", manifest)
    summaries = []
    try:
        X, y, metadata = load_partition(config["protocol"], "train")
        folds = make_folds(y, metadata, config["protocol"], count=config["cv"]["outer_folds"], seed=config["seed"])
        assignments = metadata[["track_id", "artist_key"]].reset_index(drop=True)
        assignments["hit"] = y.to_numpy()
        assignments["fold"] = -1
        for number, (_, validation) in enumerate(folds, 1):
            assignments.loc[validation, "fold"] = number
        assignments.to_csv(report_root / "outer_fold_assignments.csv", index=False)
        with threadpool_limits(limits=1):
            for name in config["family_order"]:
                summary, artifacts = train_family(name, X, y, metadata, folds, config, model_root, report_root)
                summaries.append(summary)
                manifest["artifacts"].extend(artifacts)
                manifest["completed_families"].append(name)
                pd.DataFrame(summaries).to_csv(report_root / "results.csv", index=False)
                write_json(report_root / "manifest.json", manifest)
                print(f"Completed {name}: outer CV accuracy {100*summary['cv_accuracy_mean']:.2f}% +/- {100*summary['cv_accuracy_std']:.2f}%", flush=True)
        result = pd.DataFrame(summaries)
        result["tie_order"] = result.model.map({name: i for i, name in enumerate(config["family_order"])})
        chosen = result.sort_values(["cv_accuracy_mean", "tie_order"], ascending=[False, True]).iloc[0]
        manifest["candidate"] = {"model": chosen.model, "artifact": chosen.artifact,
            "cv_accuracy_mean": float(chosen.cv_accuracy_mean), "threshold": float(chosen.final_threshold),
            "parameters": json.loads(chosen.final_parameters),
            "selection_policy": "Highest mean outer training CV accuracy; exact ties use declared family order. Development candidate only."}
        manifest.update(status="complete: training-only accuracy development", completed_at_utc=datetime.now(UTC).isoformat())
        manifest["report_hashes"] = {p.relative_to(report_root).as_posix(): sha256(p)
                                   for p in report_root.rglob("*.csv")}
        write_json(report_root / "manifest.json", manifest)
        write_json(model_root / "metadata.json", manifest)
        write_report(manifest, result)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(report_root / "manifest.json", manifest)
        raise
    return manifest


def write_report(manifest, results):
    lines = ["# Course-scoped accuracy development", "", f"Run: `{manifest['run_id']}`.",
        "Only logistic/polynomial logistic regression, decision trees, bagging, random forests and tree boosting were used.",
        "Scores are mean +/- sample SD of five outer artist-disjoint training-validation folds.",
        "Inner CV selects parameters using accuracy, then fitting-data OOF predictions choose a threshold.",
        "Outer validation labels never choose parameters or thresholds. Existing holdout sets were not loaded.", "",
        "| Model | Default CV accuracy | Threshold-tuned CV accuracy | CV ROC-AUC | Train accuracy |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in results.sort_values("cv_accuracy_mean", ascending=False).itertuples():
        lines.append(f"| {row.model} | {100*row.cv_default_accuracy_mean:.2f}% | {100*row.cv_accuracy_mean:.2f}% +/- {100*row.cv_accuracy_std:.2f}% | {row.cv_roc_auc_mean:.4f} | {100*row.train_accuracy:.2f}% |")
    candidate = manifest["candidate"]
    lines.extend(["", f"Development candidate: **{candidate['model']}**, mean outer CV accuracy **{100*candidate['cv_accuracy_mean']:.2f}%**.",
        f"Final fitting-data cutoff: `{candidate['threshold']}`. Parameters: `{json.dumps(candidate['parameters'], sort_keys=True)}`.",
        "A fresh independent test set is needed for a new final performance claim. The previously published v1 tests remain unchanged.",
        "These are exploratory development results after inspection of v1 scores; they do not certify 85–90% future accuracy.",
        f"Source/data SHA-256: `{manifest['source_sha256']}`.",
        f"Actual source snapshot, all search results, inner threshold curves, outer predictions and fold models: `reports/runs/{manifest['run_id']}/`, `models/{manifest['run_id']}/`.", ""])
    repo_path("reports/course_accuracy_v2.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="course_accuracy_v2")
    parser.add_argument("--config", default="configs/accuracy_v2.yaml")
    args = parser.parse_args()
    run(args.run_id, args.config)
