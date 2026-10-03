"""Freeze the complete CV-selected suite, then evaluate untouched test sets once.

Run: python -m src.models.finalize --run-id final_suite_v1
This module never fits estimators or selects models using test results.
"""

import argparse
from datetime import UTC, datetime
import json
import re

import joblib
import pandas as pd

from src.data.common import load_config, repo_path, sha256
from src.models.compare import compatible_manifests, require_matching_folds, verified_csv
from src.models.data import load_partition
from src.models.evaluate import prediction_scores, score_metrics
from src.models.registry import MODELS
from src.models.artifacts import save_pipeline, snapshot_sources, source_files, write_json


def completed_suite():
    latest = {}
    for path in repo_path("reports/runs").glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        model = manifest.get("model")
        if model not in MODELS or not manifest["status"].startswith("complete"):
            continue
        if model not in latest or manifest["created_at_utc"] > latest[model][1]["created_at_utc"]:
            latest[model] = (path.parent, manifest)
    missing = set(MODELS) - set(latest)
    if missing:
        raise ValueError(f"Finish every model before final tests: {sorted(missing)}")
    base_root, base = latest["logistic_regression"]
    base_oof = verified_csv(base_root, base, "training_oof_predictions.csv")
    tables = []
    for model, (root, manifest) in latest.items():
        if not compatible_manifests(base, manifest):
            raise ValueError(f"Incompatible model run: {model}")
        require_matching_folds(base_oof, base["model"],
            verified_csv(root, manifest, "training_oof_predictions.csv"), model)
        results = verified_csv(root, manifest, "results.csv")
        eligible = results.model.eq(model) & ~results.feature_set.isin(manifest["config"]["diagnostic_feature_sets"])
        if model == "logistic_regression":
            eligible |= results.model.eq("dummy")
        tables.append(results.loc[eligible].assign(run_id=manifest["run_id"]))
    return latest, pd.concat(tables, ignore_index=True)


def choose_candidate(results, config):
    eligible = results.loc[results.protocol.eq(config["primary_protocol"])
        & results.model.isin(MODELS)
        & ~results.feature_set.isin(config["diagnostic_feature_sets"])].copy()
    if eligible.empty:
        raise ValueError("No eligible training-validation candidate")
    best = float(eligible.cv_roc_auc_mean.max())
    eligible = eligible.loc[eligible.cv_roc_auc_mean >= best - config["simplicity_auc_margin"]]
    eligible["simplicity"] = eligible.model.map({model: i for i, model in enumerate(config["model_simplicity_order"])})
    if eligible.simplicity.isna().any():
        raise ValueError("Missing predefined model simplicity rank")
    eligible["features"] = eligible.feature_set.map(lambda name: len(config["feature_sets"][name]))
    choice = eligible.sort_values(["simplicity", "features", "cv_roc_auc_mean"],
                                 ascending=[True, True, False]).iloc[0]
    return {"model": choice.model, "feature_set": choice.feature_set,
        "protocol": choice.protocol, "run_id": choice.run_id, "artifact": choice.artifact,
        "cv_roc_auc_mean": float(choice.cv_roc_auc_mean), "best_cv_roc_auc_mean": best,
        "policy": f"Artist-disjoint nested CV ROC-AUC within {config['simplicity_auc_margin']} of best; predefined family simplicity order, then fewer inputs, then larger CV AUC.",
        "simplicity_order": config["model_simplicity_order"]}


def evaluate_frozen(selection, output, *, models=None):
    """Require the exact selection file before requesting any test partition."""
    frozen = output / "selection.json"
    canonical = json.loads(json.dumps(selection, allow_nan=False))
    if not frozen.is_file() or json.loads(frozen.read_text(encoding="utf-8")) != canonical:
        raise ValueError("Write and verify the frozen selection before test access")
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(frozen) != recorded["selection_sha256"]:
            raise ValueError("Frozen selection checksum changed")
    if "source_sha256" in selection:
        source = repo_path("data/interim/model_table.parquet")
        splits = repo_path("data/processed/user2_splits.csv")
        if sha256(source) != selection["source_sha256"] or sha256(splits) != selection["splits_sha256"]:
            raise ValueError("Frozen evaluation data or split assignments changed")
    metrics, predictions = [], []
    for protocol in selection["protocols"]:
        X, y, metadata = load_partition(protocol, "test", allow_test=True)
        for item in selection["artifacts"]:
            if item["protocol"] != protocol or (models is not None and item["model"] not in models):
                continue
            path = repo_path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Frozen model changed: {path}")
            pipeline = joblib.load(path)
            scores, kind = prediction_scores(pipeline, X)
            predicted = pipeline.predict(X)
            identity = {key: item[key] for key in ("protocol", "model", "feature_set", "run_id")}
            chosen = selection["chosen"]
            is_chosen = all(identity[key] == chosen[key] for key in ("protocol", "model", "feature_set", "run_id"))
            metrics.append({**identity, "selected_before_test": is_chosen,
                "test_rows": len(y), "score_kind": kind,
                **score_metrics(y, scores, predictions=predicted, score_kind=kind)})
            frame = metadata[["track_id", "artist_key", "artist_name", "title"]].reset_index(drop=True).copy()
            frame["hit"] = y.to_numpy()
            frame["score"] = scores
            frame["score_kind"] = kind
            frame["prediction"] = predicted
            frame["correct"] = predicted == y.to_numpy()
            predictions.append(frame.assign(**identity))
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def write_report(selection, results, comparison):
    chosen = selection["chosen"]
    lines = ["# Frozen model suite: final held-out evaluation", "",
        f"Chosen before opening either test set: **{chosen['model']} / {chosen['feature_set']}**.",
        f"Selection rule: {chosen['policy']}",
        f"Chosen training CV AUC: {chosen['cv_roc_auc_mean']:.4f}; best suite CV AUC: {chosen['best_cv_roc_auc_mean']:.4f}.", "",
        "The table evaluates frozen models; a larger test score does not replace the model selected in advance.",
        "Artist-disjoint is the primary protocol. The two protocols share data and are not independent replications.", "",
        "| Protocol | Model | Features | Selected in advance | Test accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results.sort_values(["protocol", "model", "feature_set"]).itertuples():
        lines.append(f"| {row.protocol} | {row.model} | {row.feature_set} | {'yes' if row.selected_before_test else ''} | "
            f"{row.accuracy:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.roc_auc:.4f} | {row.average_precision:.4f} |")
    selected = results.loc[results.selected_before_test].iloc[0]
    lines.extend(["", "## Frozen hyperparameters", "",
        "These values were selected on full training CV before either holdout was opened.",
        "All fixed estimator parameters are also saved in `reports/frozen_hyperparameters.json` and each model bundle.", "",
        "| Protocol | Model | Features | Tuned parameters |", "| --- | --- | --- | --- |"])
    for item in selection["artifacts"]:
        if item["model"] not in MODELS:
            continue
        keys = [key[:-5] for key in selection["config"][item["model"]] if key.endswith("_grid")]
        parameters = {key: item["estimator_parameters"][key] for key in keys}
        lines.append(f"| {item['protocol']} | {item['model']} | {item['feature_set']} | `{json.dumps(parameters, sort_keys=True)}` |")
    lines.extend(["", "## Selected model evidence", "",
        f"Held-out confusion matrix: TN={selected.tn}, FP={selected.fp}, FN={selected.fn}, TP={selected.tp}.",
        f"Saved model: `models/best_pipeline.joblib`; metadata: `models/metadata.json`.",
        f"Frozen selection, all metrics, predictions, actual testing source and hashes: `reports/runs/{selection['run_id']}/`.",
        "Combined train/CV/test metrics: `reports/model_comparison.csv`.",
        "ROC, precision-recall, confusion matrix and probability reliability figures: `reports/figures/final_suite_v1/`.", "",
        "## Limits", "",
        "Raw source files are absent, so source labels and the full extraction cannot be independently reproduced locally.",
        "This is preliminary four-input Mode B evidence, not numerical reproduction of the original study.",
        "The balanced research sample does not yield natural-release commercial success probabilities.",
        "SVM scores are uncalibrated margins; their log loss and Brier scores are intentionally absent.",
        "Final pipelines were trained only on their original training partitions. No holdout retraining or test-guided selection occurred.", ""])
    repo_path("reports/final_model_results.md").write_text("\n".join(lines), encoding="utf-8")


def run(run_id="final_suite_v1"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid final run ID")
    output = repo_path(f"reports/runs/{run_id}")
    if output.exists() or repo_path("models/best_pipeline.joblib").exists():
        raise FileExistsError("Final evaluation already started or model published; preserve it instead of repeating test selection")
    config = load_config("configs/models.yaml")
    suite, cv_results = completed_suite()
    chosen = choose_candidate(cv_results, config)
    artifacts = []
    for row in cv_results.itertuples():
        manifest = suite["logistic_regression" if row.model == "dummy" else row.model][1]
        item = next(item for item in manifest["artifacts"] if item["path"] == row.artifact)
        if sha256(repo_path(item["path"])) != item["sha256"]:
            raise ValueError("Model artifact integrity failed")
        estimator = joblib.load(repo_path(item["path"])).named_steps["model"]
        artifacts.append({**item, "run_id": row.run_id, "estimator_parameters": estimator.get_params(deep=False)})
    baseline = suite["logistic_regression"][1]
    output.mkdir(parents=True)
    selection = {"run_id": run_id, "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": baseline["source_sha256"], "splits_sha256": baseline["splits_sha256"],
        "protocols": config["protocols"], "chosen": chosen, "artifacts": artifacts,
        "config": config, "test_sets_evaluated_at_freeze": False}
    write_json(output / "selection.json", selection)
    write_json(repo_path("reports/frozen_hyperparameters.json"), {
        "final_run_id": run_id, "chosen_before_testing": chosen,
        "models": [{key: item[key] for key in ("model", "protocol", "feature_set", "features",
                  "estimator_parameters", "threshold", "path", "sha256")} for item in artifacts]})
    source_hashes = snapshot_sources(source_files(repo_path(".")), output)
    manifest = {"run_id": run_id, "status": "running: frozen final evaluation",
                "selection_sha256": sha256(output / "selection.json"), "source_hashes": source_hashes}
    write_json(output / "manifest.json", manifest)
    return finish_frozen(selection, output, manifest, cv_results)


def finish_frozen(selection, output, manifest, cv_results):
    chosen, artifacts, run_id = selection["chosen"], selection["artifacts"], selection["run_id"]
    try:
        results, predictions = evaluate_frozen(selection, output)
        results.to_csv(output / "test_metrics.csv", index=False)
        predictions.to_csv(output / "test_predictions.csv", index=False)
        comparison = cv_results.merge(results, on=["protocol", "model", "feature_set", "run_id"], validate="one_to_one")
        comparison.to_csv(repo_path("reports/model_comparison.csv"), index=False)
        item = next(item for item in artifacts if item["path"] == chosen["artifact"])
        X, _, _ = load_partition(chosen["protocol"], "train")
        evidence = save_pipeline(joblib.load(repo_path(chosen["artifact"])), X.iloc[:32], repo_path("models/best_pipeline.joblib"))
        test_row = results.loc[results.selected_before_test].iloc[0]
        metadata = {"selection": chosen, "features": item["features"], "threshold": item["threshold"],
            "units": {"tempo": "beats per minute", "loudness": "dB", "duration": "seconds", "artist_score": "binary 0/1"},
            "probability_interpretation": "Balanced research-sample probability; does not estimate natural-release commercial success.",
            "estimator_parameters": item["estimator_parameters"], "source_sha256": selection["source_sha256"],
            "splits_sha256": selection["splits_sha256"], "final_run_id": run_id,
            "training_rows": item["training_rows"], "test_metrics": {key: float(test_row[key]) for key in
                ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")}, **evidence}
        write_json(repo_path("models/metadata.json"), metadata)
        from src.models.plots import plot_final
        plot_final(selection, results, predictions)
        write_report(selection, results, comparison)
        manifest.update(status="complete: frozen final evaluation", completed_at_utc=datetime.now(UTC).isoformat(),
            report_hashes={path.name: sha256(path) for path in output.glob("*.csv")},
            best_pipeline_sha256=evidence["sha256"])
        write_json(output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed: frozen final evaluation", error=str(error))
        write_json(output / "manifest.json", manifest)
        raise
    print(results.to_string(index=False))
    print(f"Selected before testing: {chosen['model']}/{chosen['feature_set']}")
    return selection


def resume_before_test(run_id):
    """Recover only the known serialization failure that precedes test access."""
    output = repo_path(f"reports/runs/{run_id}")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    error = "Write and verify the frozen selection before test access"
    if (manifest["status"] != "failed: frozen final evaluation" or manifest.get("error") != error
            or (output / "test_metrics.csv").exists() or repo_path("models/best_pipeline.joblib").exists()):
        raise ValueError("Resume is allowed only for the known pre-test serialization failure")
    frozen = output / "selection.json"
    if sha256(frozen) != manifest["selection_sha256"]:
        raise ValueError("Original frozen selection changed")
    selection = json.loads(frozen.read_text(encoding="utf-8"))
    _, cv_results = completed_suite()
    if set(cv_results.artifact) != {item["path"] for item in selection["artifacts"]}:
        raise ValueError("Frozen suite no longer matches completed training runs")
    sources = [path.relative_to(repo_path(".")).as_posix() for folder in ("src", "configs", "tests")
               for path in repo_path(folder).rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".md"}]
    manifest["recovery_source_hashes"] = snapshot_sources(
        [*sources, "requirements.txt", "requirements-lock.txt"], output / "recovery_sources")
    manifest["recovery"] = {"original_error": manifest.pop("error"),
        "resumed_at_utc": datetime.now(UTC).isoformat(), "test_access_before_recovery": False,
        "selection_unchanged": True}
    manifest["status"] = "running: resumed frozen final evaluation"
    write_json(path, manifest)
    return finish_frozen(selection, output, manifest, cv_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final_suite_v1")
    parser.add_argument("--resume-before-test", action="store_true", help="Recover the known pre-test JSON failure without reselection")
    args = parser.parse_args()
    (resume_before_test if args.resume_before_test else run)(args.run_id)
