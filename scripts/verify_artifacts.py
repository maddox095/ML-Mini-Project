"""Read-only verification of saved datasets, runs, models and all code bundles.

Run from any folder: python scripts/verify_artifacts.py
This never fits models, reads test rows or changes previous artifacts.
"""

import hashlib
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
        splits = (ROOT / "data/processed/user2_splits.csv").read_bytes()
        if hashlib.sha256(splits).hexdigest() != manifest["splits_sha256"]:
            # Historical runs hashed CRLF bytes. Require identical assignments
            # after the repository's LF migration; retain original run records.
            original = splits.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if hashlib.sha256(original).hexdigest() != manifest["splits_sha256"]:
                raise ValueError("Saved split assignments changed")
    else:
        archived = ROOT / "models/deployments" / folder.name / "best_pipeline.joblib"
        verify_file(archived if archived.exists() else ROOT / "models/best_pipeline.joblib",
                    manifest["best_pipeline_sha256"])
        verify_file(folder / "selection.json", manifest["selection_sha256"])


def verify_bundles(index_path):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index["bundles"]:
        archive = ROOT / entry["archive"]
        verify_file(archive, entry.get("archive_sha256", entry.get("sha256")))
        with zipfile.ZipFile(archive) as saved:
            if saved.testzip() is not None:
                raise ValueError(f"Corrupt archive: {archive}")
            record = json.loads(saved.read("bundle_manifest.json"))
            for name, expected in record["file_hashes"].items():
                if hashlib.sha256(saved.read(name)).hexdigest() != expected:
                    raise ValueError(f"Saved bundle member checksum changed: {archive}/{name}")
            original_splits = saved.read("data/processed/user2_splits.csv")
            current_splits = (ROOT / "data/processed/user2_splits.csv").read_bytes()
            if original_splits.replace(b"\r\n", b"\n") != current_splits:
                raise ValueError(f"Saved bundle split assignments changed: {archive}")
        print(f"Verified {entry['model']}: complete ZIP and recorded files", flush=True)
    return len(index["bundles"])


def main():
    active = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    verify_file(ROOT / "models/best_pipeline.joblib", active["sha256"])
    verify_file(ROOT / active["selection"]["artifact"], active["sha256"])
    # Expanded run directories are optional; their records are also in ZIPs.
    for run_id, course in [("final_suite_v1", False), ("course_accuracy_v2", True)]:
        folder = ROOT / "reports/runs" / run_id
        if folder.is_dir():
            verify_run(folder, course=course)
    count = verify_bundles(ROOT / "models/reproducibility/index.json")
    count += verify_bundles(ROOT / "models/reproducibility_v2/course_accuracy_v2/index.json")
    print(f"Verified all {count} saved bundles, split assignments and available local runs.")


if __name__ == "__main__":
    main()
