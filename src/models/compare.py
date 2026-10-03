"""Compare completed training-only runs with matching data and outer folds."""

import json

import pandas as pd

from src.data.common import repo_path, sha256


def compatible_manifests(left: dict, right: dict) -> bool:
    for key in ("source_sha256", "splits_sha256", "dataset_version"):
        if left[key] != right[key]:
            return False
    for key in ("seed", "protocols", "primary_protocol", "feature_sets"):
        if left["config"][key] != right["config"][key]:
            return False
    for key in ("outer_folds", "inner_folds", "selection_metric"):
        if left["config"]["cv"][key] != right["config"]["cv"][key]:
            return False
    return not left["test_sets_evaluated"] and not right["test_sets_evaluated"]


def outer_assignments(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = predictions.loc[predictions.model.eq(model)].copy()
    keys = ["protocol", "track_id"]
    columns = [*keys, "artist_key", "hit", "fold"]
    for _, feature_set in subset.groupby("feature_set"):
        if feature_set.duplicated(keys).any():
            raise ValueError("Repeated identities in out-of-fold predictions")
    if subset.groupby(keys)[["artist_key", "hit", "fold"]].nunique().gt(1).any().any():
        raise ValueError("Feature comparisons use inconsistent validation assignments")
    return subset[columns].drop_duplicates(keys).sort_values(keys).reset_index(drop=True)


def require_matching_folds(left, left_model: str, right, right_model: str) -> None:
    a = outer_assignments(left, left_model)
    b = outer_assignments(right, right_model)
    if not a.equals(b):
        raise ValueError("Model comparisons must use identical training-validation identities, labels and folds")


def verified_csv(root, manifest: dict, name: str) -> pd.DataFrame:
    path = root / name
    if sha256(path) != manifest["report_hashes"][name]:
        raise ValueError(f"Stored run report changed: {name}")
    return pd.read_csv(path)


def write_comparison() -> None:
    latest = {}
    for path in repo_path("reports/runs").glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest["status"].startswith("complete") or "model" not in manifest:
            continue
        model = manifest["model"]
        if model not in latest or manifest["created_at_utc"] > latest[model][1]["created_at_utc"]:
            latest[model] = (path.parent, manifest)
    if "logistic_regression" not in latest:
        return
    base_root, baseline = latest["logistic_regression"]
    base_oof = verified_csv(base_root, baseline, "training_oof_predictions.csv")
    tables, run_ids, excluded = [], [], []
    for model, (root, manifest) in sorted(latest.items()):
        if not compatible_manifests(baseline, manifest):
            excluded.append(manifest["run_id"])
            continue
        predictions = verified_csv(root, manifest, "training_oof_predictions.csv")
        require_matching_folds(base_oof, baseline["model"], predictions, model)
        results = verified_csv(root, manifest, "results.csv")
        results = results.loc[results.model.eq(model)
            & ~results.feature_set.isin(manifest["config"]["diagnostic_feature_sets"])].copy()
        tables.append(results.assign(run_id=manifest["run_id"]))
        run_ids.append(manifest["run_id"])
    results = pd.concat(tables, ignore_index=True)
    lines = ["# Model comparison in training validation", "",
        "These runs use matching dataset/split hashes, CV settings, training identities and outer validation folds.",
        "Metrics are nested cross-validation means +/- sample standard deviations. These training runs did not evaluate final test sets; see [the separate final evaluation](final_model_results.md).", "",
        "| Protocol | Model | Inputs | CV accuracy | CV ROC-AUC | CV F1 |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in results.sort_values(["protocol", "feature_set", "model"]).itertuples():
        lines.append(f"| {row.protocol} | {row.model} | {row.feature_set} | "
            f"{row.cv_accuracy_mean:.3f} +/- {row.cv_accuracy_std:.3f} | "
            f"{row.cv_roc_auc_mean:.3f} +/- {row.cv_roc_auc_std:.3f} | {row.cv_f1_mean:.3f} |")
    primary = results.loc[results.protocol.eq(baseline["config"]["primary_protocol"])
                          & results.feature_set.eq("audio_artist_score")]
    lr = primary.loc[primary.model.eq("logistic_regression")]
    if not lr.empty:
        reference = lr.iloc[0]
        lines.extend(["", "## Artist-disjoint comparison with all four inputs", ""])
        for row in primary.loc[~primary.model.eq("logistic_regression")].itertuples():
            lines.append(f"- {row.model} versus LR: accuracy difference "
                f"{100 * (row.cv_accuracy_mean - reference.cv_accuracy_mean):+.2f} percentage points; "
                f"ROC-AUC difference {row.cv_roc_auc_mean - reference.cv_roc_auc_mean:+.4f}.")
        lines.append("These differences support later model selection; they are not evidence of a final-test winner.")
    lines.extend(["", "## Provenance", "",
        f"- Dataset SHA-256: `{baseline['source_sha256']}`.",
        f"- Split SHA-256: `{baseline['splits_sha256']}`.",
        f"- Included runs: {', '.join(f'`{run_id}`' for run_id in run_ids)}.",
        "- Artist-Score-only results are diagnostic and remain in each model's individual report.",
        "- Probability-quality metrics are available for LR, trees, forests and NN; uncalibrated SVM margins are not probabilities.",
        "- Raw sources remain absent locally; this is preliminary Mode B methodology evidence.", ""])
    if excluded:
        lines.append(f"Excluded runs with different data/validation settings: {', '.join(excluded)}.")
    repo_path("reports/model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_comparison()
