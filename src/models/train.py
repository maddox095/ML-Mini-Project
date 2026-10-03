"""Train an implemented model using training-only nested CV.

Run: python -m src.models.train
Final test partitions are never requested by this module.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
import platform
import re
import subprocess
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data.common import load_config, repo_path, sha256
from src.data.verify_handoff import verify_handoff
from src.features.build_features import feature_order, make_pipeline
from src.models.data import PROTOCOLS, load_partition
from src.models.evaluate import METRICS, prediction_scores, score_metrics
from src.models.registry import MODELS, TITLES, TREE_MODELS, make_estimator, search_space
from src.models.artifacts import save_pipeline, source_files, write_json
from src.models.artifacts import snapshot_sources as _snapshot_sources
from src.models.validation import fit_candidate, make_folds


def snapshot_sources(names, root):
    """Keep the existing public entry point and repository resolver compatible."""
    return _snapshot_sources(names, root, path_resolver=repo_path)


def train_experiment(name: str, features: list[str], X, y, metadata,
                     protocol: str, folds: list, config: dict) -> dict:
    """Evaluate fold-specific tuning on separate outer validation rows."""
    pipeline = make_pipeline(make_estimator(name, config), features=features, scale=name not in TREE_MODELS)
    grid = search_space(name, config)
    scores = np.full(len(y), np.nan)
    predicted = np.full(len(y), -1, dtype=int)
    fold_number = np.full(len(y), -1, dtype=int)
    fold_results, searches = [], []
    started = perf_counter()
    for fold, (training, validation) in enumerate(folds, start=1):
        inner = make_folds(y.iloc[training], metadata.iloc[training], protocol,
                           count=config["cv"]["inner_folds"], seed=config["seed"] + fold)
        fitted, tuning, seconds = fit_candidate(
            pipeline, grid, X.iloc[training], y.iloc[training], inner, config)
        scores[validation], score_kind = prediction_scores(fitted, X.iloc[validation])
        predicted[validation] = fitted.predict(X.iloc[validation])
        fold_number[validation] = fold
        metrics = score_metrics(y.iloc[validation], scores[validation],
                               predictions=predicted[validation], score_kind=score_kind)
        fold_results.append({"fold": fold, "train_rows": len(training),
            "validation_rows": len(validation), "fit_seconds": seconds,
            "selected_parameters": json.dumps({key: fitted.get_params()[key] for key in
                (grid if isinstance(grid, dict) else grid[0])}, sort_keys=True),
            "selected_C": getattr(fitted.named_steps["model"], "C", np.nan), **metrics})
        if not tuning.empty:
            searches.append(tuning.assign(stage=f"outer_fold_{fold}"))
    if not np.isfinite(scores).all() or (fold_number < 1).any():
        raise ValueError("Some training rows have no out-of-fold prediction")
    final_folds = make_folds(y, metadata, protocol, count=config["cv"]["inner_folds"],
                             seed=config["seed"])
    fitted, tuning, _ = fit_candidate(pipeline, grid, X, y, final_folds, config)
    if not tuning.empty:
        searches.append(tuning.assign(stage="full_training_tuning"))
    fold_frame = pd.DataFrame(fold_results)
    train_scores, score_kind = prediction_scores(fitted, X)
    train_metrics = score_metrics(y, train_scores, predictions=fitted.predict(X), score_kind=score_kind)
    summary = {"train_rows": len(y), "score_kind": score_kind,
               "selected_C": getattr(fitted.named_steps["model"], "C", np.nan),
               "fit_seconds": perf_counter() - started, "train_accuracy": train_metrics["accuracy"]}
    best_parameters = {key: fitted.get_params()[key] for key in
                       (grid if isinstance(grid, dict) else grid[0])}
    summary["selected_parameters"] = json.dumps(best_parameters, sort_keys=True)
    for metric in METRICS:
        if metric not in fold_frame:
            continue
        summary[f"cv_{metric}_mean"] = float(fold_frame[metric].mean())
        summary[f"cv_{metric}_std"] = float(fold_frame[metric].std(ddof=1))
    summary.update({f"cv_{key}": int(fold_frame[key].sum()) for key in ("tn", "fp", "fn", "tp")})
    oof = metadata[["track_id", "artist_key"]].reset_index(drop=True).copy()
    oof["hit"] = y.to_numpy()
    oof["fold"] = fold_number
    oof[score_kind] = scores
    oof["score_kind"] = score_kind
    oof["prediction"] = predicted
    return {"pipeline": fitted, "summary": summary, "folds": fold_frame,
            "searches": pd.concat(searches, ignore_index=True) if searches else pd.DataFrame(),
            "oof": oof}


def robustness_checks(pipeline, X, y, metadata, protocol: str, config: dict) -> pd.DataFrame:
    """Compare fixed selected parameters across CV seeds and shuffled labels."""
    rows = []
    for seed in config["robustness_seeds"]:
        folds = make_folds(y, metadata, protocol, count=config["cv"]["outer_folds"], seed=seed)
        for shuffled in (False, True):
            for fold, (training, validation) in enumerate(folds, start=1):
                labels = y.iloc[training].to_numpy().copy()
                if shuffled:
                    labels = np.random.default_rng(seed + fold).permutation(labels)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    fitted = clone(pipeline).fit(X.iloc[training], labels)
                values, score_kind = prediction_scores(fitted, X.iloc[validation])
                metrics = score_metrics(y.iloc[validation], values,
                    predictions=fitted.predict(X.iloc[validation]), score_kind=score_kind)
                rows.append({"check": "shuffled_training_labels" if shuffled else "fixed_parameters_seed_stability",
                             "seed": seed, "fold": fold, **metrics})
    return pd.DataFrame(rows)


def write_summary(manifest: dict, results: pd.DataFrame, robustness: pd.DataFrame) -> None:
    model = manifest["model"]
    title = TITLES[model]
    lines = [f"# {title} baseline", "",
        f"Run: `{manifest['run_id']}`. Device: CPU. Status: preliminary Mode B experiment.", "",
        f"Scores below are mean +/- sample standard deviation across {manifest['config']['cv']['outer_folds']} outer training-validation folds.",
        f"Each {title} fold tunes its parameters in {manifest['config']['cv']['inner_folds']} separate inner folds. The final test sets were not evaluated.", "",
        "| Protocol | Model | Features | CV accuracy | CV ROC-AUC | CV F1 | Selected C |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in results.itertuples():
        c = "-" if pd.isna(row.selected_C) else f"{row.selected_C:g}"
        lines.append(f"| {row.protocol} | {row.model} | {row.feature_set} | "
            f"{row.cv_accuracy_mean:.3f} +/- {row.cv_accuracy_std:.3f} | "
            f"{row.cv_roc_auc_mean:.3f} +/- {row.cv_roc_auc_std:.3f} | {row.cv_f1_mean:.3f} | {c} |")
    candidate = manifest["candidate"]
    lines.extend(["", "## Candidate for later comparison", "",
        f"The current {title} candidate uses `{candidate['feature_set']}` under the artist-disjoint protocol.",
        f"Pipeline: `{candidate['artifact']}`. This is a baseline candidate, not the final model for deployment.",
        "Final refit parameters are selected by CV on all outer training rows; outer folds tune independently.", "",
        "## Training-only diagnostics", ""])
    for check, group in robustness.groupby("check", sort=False):
        seed_means = group.groupby("seed")[["accuracy", "roc_auc"]].mean()
        lines.append(f"- {check}: mean accuracy {seed_means.accuracy.mean():.3f}; mean ROC-AUC "
                     f"{seed_means.roc_auc.mean():.3f}, SD across seed means {seed_means.roc_auc.std(ddof=1):.4f}.")
    lines.extend(["", f"All {len(manifest['artifacts'])} saved pipelines passed save/reload score and class agreement checks.",
        "Audio-only and Artist-Score-only comparisons quantify the influence of artist history.",
        "These are training-validation results; CV-guided model/feature selection still requires final held-out evaluation.", "",
        "## Provenance and limitations", "",
        f"- Dataset SHA-256: `{manifest['source_sha256']}`.",
        f"- Split SHA-256: `{manifest['splits_sha256']}`.",
        f"- Full metrics, fold assignments, searches, coefficients, predictions and manifest: `{manifest['reports_directory']}`.",
        "- Feature coefficients/importances, where available, are associations rather than causal effects.",
        "- Raw sources remain absent locally; source-label reproducibility and exact numerical reproduction are not established.",
        "- Balanced sampling probabilities do not estimate commercial success probabilities for natural music releases.", "",
        f"Reproduce with `python -m src.models.train --model {model}`; each run creates separate artifact directories.", ""])
    if model == "svm_linear":
        lines.extend(["## Implementation choice", "",
            "This linear SVM uses `LinearSVC`, L2 regularization, squared-hinge loss and a primal solver.",
            "Its loss and intercept treatment differ from `SVC(kernel='linear')`; this choice is recorded as a methodology adaptation.",
            "ROC-AUC and average precision use raw decision margins. Class predictions use the zero-margin boundary.",
            "Margins are not probabilities. Probability calibration is deferred; log loss and Brier scores are omitted for SVM rows.", "",
            "Method reference: [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).", ""])
    elif model in {"svm_rbf", "svm_polynomial"}:
        lines.extend(["Kernel SVMs use native decision margins, not calibrated probabilities.",
            "Log loss and Brier score are omitted for the uncalibrated SVM.", ""])
    elif model == "neural_network":
        lines.extend(["The network has one hidden layer of six sigmoid units and a binary sigmoid output with L2 regularization.",
            "L-BFGS replaces the proposed Adam/epoch-checkpoint schedule. There is no internal random validation split.",
            "The optimizer and iteration budget are recorded methodology adaptations; iterations are not Adam epochs.", ""])
    elif model == "logistic_regression":
        lines.extend(["Method reference: [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).", ""])
    report_name = "linear_svm" if model == "svm_linear" else model
    repo_path(f"reports/{report_name}_baseline.md").write_text("\n".join(lines), encoding="utf-8")


def run(*, config_path: str = "configs/models.yaml", run_id: str | None = None,
        model: str | None = None) -> dict:
    config = load_config(config_path)
    if model is not None:
        config["model"] = model
    if config["model"] not in MODELS or config["decision_threshold"] != 0.5:
        raise ValueError("Select an implemented model; probability models use the fixed 0.5 threshold")
    verification = verify_handoff()
    if not verification["integrity_passed"]:
        raise ValueError("Handoff integrity failed; resolve the verifier findings before training")
    split_manifest = json.loads(repo_path("data/processed/user2_split_manifest.json").read_text(encoding="utf-8"))
    run_id = run_id or f"{config['model']}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{verification['input_sha256'][:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run ID must contain only letters, numbers, underscores or hyphens")
    report_root = repo_path(f"reports/runs/{run_id}")
    model_root = repo_path(f"models/{run_id}")
    if report_root.exists() or model_root.exists():
        raise FileExistsError("Run already exists; choose a new run ID instead of overwriting artifacts")
    if config["primary_protocol"] not in config["protocols"]:
        raise ValueError("Primary protocol must be included in protocols")
    for protocol in config["protocols"]:
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown protocol: {protocol}")
    for features in config["feature_sets"].values():
        feature_order(features=features)
    report_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path("."),
                            capture_output=True, text=True, check=False)
    sources = source_files(repo_path("."))
    manifest = {"run_id": run_id, "status": "running", "model": config["model"],
        "device": "cpu", "test_sets_evaluated": False, "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": verification["dataset_version"], "source_sha256": verification["input_sha256"],
        "splits_sha256": split_manifest["splits_sha256"], "config": config,
        "source_hashes": snapshot_sources(sources, report_root),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow", "threadpoolctl"]},
        "source_review": {"raw_inputs_verified": verification["raw_inputs_verified"],
                          "limitations": verification["limitations"]},
        "reports_directory": report_root.relative_to(repo_path(".")).as_posix(), "artifacts": []}
    write_json(report_root / "manifest.json", manifest)
    summaries, folds_all, searches_all, oof_all, coefficients = [], [], [], [], []
    loaded_training = {}
    try:
        with threadpool_limits(limits=1):
            for protocol in config["protocols"]:
                X, y, metadata = load_partition(protocol, "train")
                loaded_training[protocol] = (X, y, metadata)
                folds = make_folds(y, metadata, protocol, count=config["cv"]["outer_folds"], seed=config["seed"])
                experiments = [("dummy", "prior", feature_order()),
                    *[(config["model"], label, feature_order(features=features))
                      for label, features in config["feature_sets"].items()]]
                for name, label, features in experiments:
                    print(f"Training {protocol}: {name}/{label} on {len(y):,} rows", flush=True)
                    result = train_experiment(name, features, X, y, metadata, protocol, folds, config)
                    artifact = model_root / f"{protocol}__{name}__{label}.joblib"
                    artifact_check = save_pipeline(result["pipeline"], X.iloc[:32], artifact)
                    artifact_name = artifact.relative_to(repo_path(".")).as_posix()
                    identity = {"protocol": protocol, "model": name, "feature_set": label}
                    summaries.append({**identity, **result["summary"], "artifact": artifact_name})
                    folds_all.append(result["folds"].assign(**identity))
                    oof_all.append(result["oof"].assign(**identity))
                    if not result["searches"].empty:
                        searches_all.append(result["searches"].assign(**identity))
                    estimator = result["pipeline"].named_steps["model"]
                    if name in MODELS and (hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_")):
                        names = result["pipeline"].named_steps["preprocessor"].get_feature_names_out()
                        values = estimator.coef_[0] if hasattr(estimator, "coef_") else estimator.feature_importances_
                        coefficients.extend({**identity, "transformed_feature": feature, "coefficient": float(value)}
                                            for feature, value in zip(names, values))
                    manifest["artifacts"].append({**identity, "path": artifact_name, "features": features,
                        "estimator_parameters": estimator.get_params(deep=False),
                        "optimizer_iterations": int(estimator.n_iter_) if name == "neural_network" else None,
                        "optimizer_loss": float(estimator.loss_) if name == "neural_network" else None,
                        "training_rows": len(y),
                        "threshold": 0.5 if artifact_check["score_kind"] == "probability" else 0.0,
                        **artifact_check})
            results = pd.DataFrame(summaries)
            eligible = results.loc[(results.protocol == config["primary_protocol"])
                & (results.model == config["model"])
                & ~results.feature_set.isin(config["diagnostic_feature_sets"])].copy()
            best_auc = eligible.cv_roc_auc_mean.max()
            eligible = eligible.loc[eligible.cv_roc_auc_mean >= best_auc - config["simplicity_auc_margin"]]
            eligible["feature_count"] = eligible.feature_set.map(lambda label: len(config["feature_sets"][label]))
            candidate = eligible.sort_values(["feature_count", "cv_roc_auc_mean"], ascending=[True, False]).iloc[0]
            manifest["candidate"] = {"feature_set": candidate.feature_set, "artifact": candidate.artifact,
                "selected_C": None if pd.isna(candidate.selected_C) else float(candidate.selected_C),
                "selected_parameters": json.loads(candidate.selected_parameters),
                "selection_protocol": config["primary_protocol"],
                "cv_roc_auc_mean": float(candidate.cv_roc_auc_mean),
                "selection_policy": f"Within {config['simplicity_auc_margin']} of best nested CV ROC-AUC, prefer fewer features; baseline candidate only."}
            X, y, metadata = loaded_training[config["primary_protocol"]]
            robustness = robustness_checks(joblib.load(repo_path(candidate.artifact)), X, y, metadata,
                                          config["primary_protocol"], config)
        results.to_csv(report_root / "results.csv", index=False)
        pd.concat(folds_all, ignore_index=True).to_csv(report_root / "fold_metrics.csv", index=False)
        pd.concat(searches_all, ignore_index=True).to_csv(report_root / "search_results.csv", index=False)
        pd.concat(oof_all, ignore_index=True).to_csv(report_root / "training_oof_predictions.csv", index=False)
        pd.DataFrame(coefficients, columns=["protocol", "model", "feature_set", "transformed_feature", "coefficient"]).to_csv(report_root / "coefficients.csv", index=False)
        robustness.to_csv(report_root / "robustness.csv", index=False)
        manifest["status"] = "complete: training-only baseline"
        manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        manifest["report_hashes"] = {path.name: sha256(path) for path in sorted(report_root.glob("*.csv"))}
        write_json(report_root / "manifest.json", manifest)
        write_json(model_root / "metadata.json", manifest)
        write_summary(manifest, results, robustness)
        from src.models.compare import write_comparison
        write_comparison()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_json(report_root / "manifest.json", manifest)
        raise
    print(results[["protocol", "model", "feature_set", "cv_accuracy_mean", "cv_roc_auc_mean", "selected_C"]].to_string(index=False))
    print(f"Saved {report_root}; final test sets remain reserved", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--model", choices=sorted(MODELS), help="Override the default model family")
    parser.add_argument("--run-id", help="Unique artifact directory name; existing runs cannot be overwritten")
    args = parser.parse_args()
    run(config_path=args.config, run_id=args.run_id, model=args.model)


if __name__ == "__main__":
    main()
