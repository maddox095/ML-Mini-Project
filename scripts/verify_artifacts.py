"""Read-only verification of saved datasets, runs, models and all code bundles.

Run from any folder: python scripts/verify_artifacts.py
This never fits models, reads test rows or changes previous artifacts.
"""

import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.common import sha256


def verify_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Saved artifact checksum changed: {path}")


def verify_run(folder, *, course=False):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("report_hashes", {}).items():
        verify_file(folder / name, expected)
    for name, expected in manifest.get("source_hashes", {}).items():
        verify_file(folder / "source_snapshot" / name, expected)
    for artifact in manifest.get("artifacts", []):
        verify_file(ROOT / artifact["path"], artifact["sha256"])
    if course:
        verify_file(ROOT / "data/interim/model_table.parquet", manifest["source_sha256"])
        verify_file(ROOT / "data/processed/user2_splits.csv", manifest["splits_sha256"])
    else:
        verify_file(ROOT / "models/best_pipeline.joblib", manifest["best_pipeline_sha256"])
        verify_file(folder / "selection.json", manifest["selection_sha256"])


def verify_bundles(index_path):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index["bundles"]:
        archive = ROOT / entry["archive"]
        verify_file(archive, entry.get("archive_sha256", entry.get("sha256")))
        with zipfile.ZipFile(archive) as saved:
            if saved.testzip() is not None:
                raise ValueError(f"Corrupt archive: {archive}")
        bundle = archive.with_suffix("")
        record = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        for name, expected in record["file_hashes"].items():
            verify_file(bundle / name, expected)
        print(f"Verified {entry['model']}: complete ZIP and recorded files", flush=True)
    return len(index["bundles"])


def main():
    verify_run(ROOT / "reports/runs/final_suite_v1")
    verify_run(ROOT / "reports/runs/course_accuracy_v2", course=True)
    count = verify_bundles(ROOT / "models/reproducibility/index.json")
    count += verify_bundles(ROOT / "models/reproducibility_v2/course_accuracy_v2/index.json")
    print(f"Verified original/final runs, inputs and all {count} saved bundles.")


if __name__ == "__main__":
    main()
