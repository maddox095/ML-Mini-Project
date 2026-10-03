"""Save actual course-scoped source, each tried setting and fitted CV models."""

import argparse
import json
import shutil

import pandas as pd

from src.data.common import repo_path, sha256
from src.models.course_models import json_parameters, make_course_pipeline
from src.models.artifacts import write_json
from src.models.bundles import copy_sources, file_hashes, write_replay_scripts


def run(run_id="course_accuracy_v2"):
    root = repo_path(f"reports/runs/{run_id}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["status"].startswith("complete"):
        raise ValueError("Complete the accuracy run before packaging its results")
    output = repo_path(f"models/reproducibility_v2/{run_id}")
    if output.exists():
        raise FileExistsError("Preserve earlier bundles; export a new run ID")
    for name, expected in manifest.get("report_hashes", {}).items():
        if sha256(root / name) != expected:
            raise ValueError(f"Recorded course report changed: {name}")
    output.mkdir(parents=True)
    records = []
    for model in manifest["completed_families"]:
        destination = output / model
        destination.mkdir()
        copy_sources(destination)
        for name, expected in manifest["source_hashes"].items():
            original = root / "source_snapshot" / name
            if sha256(original) != expected:
                raise ValueError(f"Original source snapshot changed: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        for name in ["pytest.ini", "requirements.txt", "requirements-lock.txt", "data/interim/model_table.parquet",
            "data/processed/user2_splits.csv", "data/processed/user2_split_manifest.json", "data/raw/MANIFEST.csv",
            "reports/artist_score_audit.csv", "reports/feature_availability.csv", "reports/data_quality.csv"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(repo_path(name), target)
        for name in ["reports/course_accuracy_v2.md", "reports/source_restore_v2_all.json",
                     "docs/ACCURACY_IMPROVEMENT_PLAN.md"]:
            source = repo_path(name)
            if source.is_file():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        copy_root = destination / f"reports/runs/{run_id}"
        copy_root.mkdir(parents=True)
        for name in ["manifest.json", "results.csv", "outer_fold_assignments.csv"]:
            shutil.copyfile(root / name, copy_root / name)
        shutil.copytree(root / model, copy_root / model)
        shutil.copytree(root / "source_snapshot", copy_root / "source_snapshot")
        learning_curve = repo_path("reports/runs/course_learning_curve_v2")
        if learning_curve.is_dir():
            shutil.copytree(learning_curve, destination / "reports/runs/course_learning_curve_v2")
        model_folder = destination / f"models/{run_id}"
        model_folder.mkdir(parents=True)
        for entry in manifest["artifacts"]:
            if entry["model"] == model:
                source = repo_path(entry["path"])
                if sha256(source) != entry["sha256"]:
                    raise ValueError("Saved course checkpoint changed")
                shutil.copyfile(source, model_folder / source.name)
        trials = pd.read_csv(root / model / "search_results.csv")
        settings = destination / "hyperparameters"
        settings.mkdir()
        for number, row in enumerate(trials.itertuples(index=False)):
            name = f"setting_{number:04d}"
            parameters = json.loads(row.params)
            entry = {"model": model, "stage": row.stage, "parameters": parameters,
                "pipeline_configuration": json_parameters(make_course_pipeline(model, manifest["config"]).set_params(**parameters)),
                "config": manifest["config"], "source_sha256": manifest["source_sha256"], "splits_sha256": manifest["splits_sha256"],
                "cv_results": {key: value for key, value in row._asdict().items() if key.startswith(("mean_", "std_", "rank_", "split"))}}
            write_json(settings / f"{name}.json", entry)
            write_replay_scripts(settings, name, module="src.models.course_replay", function="replay")
        (destination / "evaluate_saved_model.py").write_text(
            "from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parent\nsys.path.insert(0,str(ROOT))\n"
            "from src.models.course_replay import evaluate_saved_cv\n"
            f"if __name__ == '__main__':\n    print(evaluate_saved_cv('{run_id}','{model}').to_string(index=False))\n", encoding="utf-8")
        (destination / "README.md").write_text(
            f"# {model}: course-scoped accuracy experiment\n\nRun: `{run_id}`. Actual fit, threshold-selection and validation code is under `src/`; original fit files are overlaid from the byte-verified source snapshot.\n\n"
            "Install the preserved `requirements-lock.txt` in Python 3.12, then run from this folder.\n\n"
            "- Replay one setting: `python hyperparameters/setting_0000_train.py`.\n"
            "- Score its first inner validation fold: `python hyperparameters/setting_0000_test.py`.\n"
            "- Score another inner fold: `python -m src.models.course_replay --setting hyperparameters/setting_0000.json --inner-fold 2 --output replay_fold2`.\n"
            "- Evaluate all saved outer-fold models: `python evaluate_saved_model.py`. No model is fitted and no v1 test set is loaded.\n"
            f"- Re-run the full suite: `python -m src.models.accuracy_v2 --run-id {run_id}_rerun`.\n\n"
            f"There are {len(trials)} tried parameter/stage records, each with complete parameters and runnable training/validation files.\n"
            "All five tuned outer-fold checkpoints and the final training pipeline are saved, including their selected decision thresholds. Discarded inner trial weights can be regenerated.\n"
            "These are training-validation development results after the v1 test scores were observed. A new final accuracy claim requires a genuinely untouched evaluation dataset.\n", encoding="utf-8")
        hashes = file_hashes(destination)
        write_json(destination / "bundle_manifest.json", {"model": model, "run_id": run_id,
            "settings_count": len(trials), "file_hashes": hashes, "original_source_hashes": manifest["source_hashes"]})
        archive = repo_path(shutil.make_archive(str(output / model), "zip", root_dir=destination))
        records.append({"model": model, "setting_records": len(trials),
            "archive": archive.relative_to(repo_path(".")).as_posix(), "sha256": sha256(archive)})
        print(f"Saved actual {model} code and {len(trials)} settings", flush=True)
    write_json(output / "index.json", {"run_id": run_id, "bundles": records})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="course_accuracy_v2")
    run(parser.parse_args().run_id)
