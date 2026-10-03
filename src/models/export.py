"""Create self-contained per-model code/data/settings/checkpoint bundles.

Training source snapshots are preserved byte for byte. Additional testing and
replay code is identified separately; it is never passed off as old fit code.
"""

import json
import shutil

from src.data.common import repo_path, sha256
from src.models.compare import verified_csv
from src.models.finalize import completed_suite
from src.models.registry import make_estimator
from src.models.artifacts import write_json
from src.models.bundles import copy_sources, file_hashes, write_replay_scripts


def export_model(root, manifest, selection, final_root, destination):
    if destination.exists():
        raise FileExistsError(f"Preserve existing bundle: {destination}")
    destination.mkdir(parents=True)
    copy_sources(destination)
    # Keep the exact fit implementation separately from the current replay tools.
    original = destination / "original_training_code"
    for folder in ("src", "configs"):
        shutil.copytree(destination / folder, original / folder)
    for source in (root / "source_snapshot").rglob("*"):
        if source.is_file():
            target = original / source.relative_to(root / "source_snapshot")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name, expected in manifest["source_hashes"].items():
        if sha256(original / name) != expected:
            raise ValueError(f"Original training code missing or altered: {name}")
    for name in ["pytest.ini", "requirements.txt", "requirements-lock.txt",
        "data/interim/model_table.parquet", "data/processed/user2_splits.csv",
        "data/processed/user2_split_manifest.json", "data/raw/MANIFEST.csv",
        "reports/artist_score_audit.csv", "reports/feature_availability.csv", "reports/data_quality.csv"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(repo_path(name), target)
        # The original fit tree is runnable independently, with the same input files.
        if name.startswith(("data/", "reports/")):
            original_target = original / name
            original_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, original_target)
    (original / "README.md").write_text(
        f"# Original {manifest['model']} fit code\n\n"
        "From this folder, install the preserved requirements lock and run:\n\n"
        f"`python -m src.models.train --model {manifest['model']}`\n\n"
        "The original recorded fit files are unchanged; input data and required audits are included.\n"
        "The bundle root contains the maintained hyperparameter replay and final testing entry points.\n", encoding="utf-8")
    shutil.copytree(root, destination / root.relative_to(repo_path(".")))
    model_root = repo_path(f"models/{manifest['run_id']}")
    shutil.copytree(model_root, destination / model_root.relative_to(repo_path(".")))
    shutil.copytree(final_root, destination / final_root.relative_to(repo_path(".")))
    figures = repo_path(f"reports/figures/{selection['run_id']}")
    if figures.is_dir():
        shutil.copytree(figures, destination / figures.relative_to(repo_path(".")))
    report_name = "linear_svm" if manifest["model"] == "svm_linear" else manifest["model"]
    for name in [f"reports/{report_name}_baseline.md", "reports/final_model_results.md",
                 "reports/model_comparison.csv", "reports/model_comparison.md", "reports/frozen_hyperparameters.json"]:
        if repo_path(name).is_file():
            shutil.copyfile(repo_path(name), destination / name)
    search = verified_csv(root, manifest, "search_results.csv")
    settings = destination / "hyperparameters"
    settings.mkdir()
    records = []
    relevant = search.loc[search.model.eq(manifest["model"])]
    for i, (_, row) in enumerate(relevant.iterrows()):
        params = json.loads(row.params)
        estimator = make_estimator(manifest["model"], manifest["config"])
        estimator.set_params(**{key.removeprefix("model__"): value for key, value in params.items()})
        entry = {"model": manifest["model"], "protocol": row.protocol,
            "feature_set": row.feature_set, "features": manifest["config"]["feature_sets"][row.feature_set],
            "stage": row.stage, "parameters": params, "config": manifest["config"],
            "estimator_parameters": estimator.get_params(deep=False),
            "source_sha256": manifest["source_sha256"], "splits_sha256": manifest["splits_sha256"],
            "cv_results": {key: value for key, value in row.items() if key.startswith(("mean_", "std_", "rank_", "split"))
                and value is not None and not (isinstance(value, float) and value != value)}}
        name = f"setting_{i:04d}"
        write_json(settings / f"{name}.json", entry)
        write_replay_scripts(settings, name, module="src.models.replay", function="run")
        records.append({"setting": name, "protocol": row.protocol, "feature_set": row.feature_set,
                        "stage": row.stage, "parameters": params})
    write_json(destination / "hyperparameter_index.json", {"settings": records})
    (destination / "evaluate_saved_model.py").write_text(
        "from pathlib import Path\nimport sys\nROOT = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(ROOT))\nimport json\n"
        "from src.models.finalize import evaluate_frozen\n"
        f"folder = ROOT / 'reports/runs/{selection['run_id']}'\n"
        "selection = json.loads((folder / 'selection.json').read_text(encoding='utf-8'))\n"
        "if __name__ == '__main__':\n"
        f"    metrics, predictions = evaluate_frozen(selection, folder, models={{'{manifest['model']}'}})\n"
        f"    metrics = metrics.loc[metrics.model.eq('{manifest['model']}')]\n"
        "    print(metrics.to_string(index=False))\n    metrics.to_csv(ROOT / 'replayed_test_metrics.csv', index=False)\n", encoding="utf-8")
    (destination / "README.md").write_text(
        f"# {manifest['model']} reproducibility bundle\n\n"
        f"Run: `{manifest['run_id']}`. Full current training/testing/replay files are under `src/`; byte-preserved original fit files and their dependencies are under `original_training_code/`.\n\n"
        "Install `requirements-lock.txt` in Python 3.12, then run commands from this folder.\n\n"
        f"- Retrain the family: `python -m src.models.train --model {manifest['model']}`.\n"
        "- Retrain one saved hyperparameter setting: `python hyperparameters/setting_0000_train.py`.\n"
        "- Reproduce its first inner validation fit: `python hyperparameters/setting_0000_test.py`.\n"
        "- Choose another inner fold: `python -m src.models.replay --setting hyperparameters/setting_0000.json --inner-fold 2 --output replay_fold2`.\n"
        "- Evaluate the unchanged frozen checkpoint on the original holdout: `python evaluate_saved_model.py`. This reproduces published results; do not use it to tune.\n\n"
        f"All {len(records)} setting/stage/feature/protocol combinations have JSON settings and runnable train/test files.\n"
        "Each setting includes the complete run configuration, inner-fold scores and tuning parameters.\n"
        "Full fixed and selected estimator parameters, fitted weights/preprocessing and training provenance are in model metadata and checkpoints.\n"
        "Inner trial weights were not retained; replay code regenerates them. Final fitted pipelines are retained.\n"
        "The small current dataset, splits and integrity audits are included. Original raw source archives remain unavailable.\n"
        "Original training source hashes identify byte-preserved fit files; additional evaluation/replay code is bundled separately and identified by bundle hashes.\n", encoding="utf-8")
    hashes = file_hashes(destination)
    write_json(destination / "bundle_manifest.json", {"model": manifest["model"], "run_id": manifest["run_id"],
        "settings_count": len(records), "original_training_source_hashes": manifest["source_hashes"], "file_hashes": hashes})
    return len(records)


def run(final_run_id="final_suite_v1"):
    suite, _ = completed_suite()
    final_root = repo_path(f"reports/runs/{final_run_id}")
    final_manifest = json.loads((final_root / "manifest.json").read_text(encoding="utf-8"))
    if not final_manifest["status"].startswith("complete"):
        raise ValueError("Complete final evaluation before exporting its code and results")
    selection = json.loads((final_root / "selection.json").read_text(encoding="utf-8"))
    output = repo_path("models/reproducibility")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, (root, manifest) in sorted(suite.items()):
        destination = output / model
        count = export_model(root, manifest, selection, final_root, destination)
        archive = shutil.make_archive(str(output / model), "zip", root_dir=destination)
        rows.append({"model": model, "run_id": manifest["run_id"], "setting_records": count,
                     "bundle": destination.relative_to(repo_path(".")).as_posix(),
                     "archive": str(repo_path(archive).relative_to(repo_path("."))).replace("\\", "/"),
                     "archive_sha256": sha256(repo_path(archive))})
        print(f"Saved complete {model} code bundle, {count} hyperparameter settings", flush=True)
    write_json(output / "index.json", {"bundles": rows})
    return rows


if __name__ == "__main__":
    run()
