"""Save models, complete source snapshots and strict experiment JSON."""

import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from src.data.common import repo_path, sha256
from src.models.evaluate import prediction_scores


def write_json(path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_pipeline(pipeline, sample, path) -> dict:
    """Round-trip native scores and classes on training examples only."""
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    original, score_kind = prediction_scores(pipeline, sample)
    reloaded, restored_kind = prediction_scores(restored, sample)
    if (score_kind != restored_kind or not np.allclose(original, reloaded, atol=1e-12, rtol=0)
            or not np.array_equal(pipeline.predict(sample), restored.predict(sample))):
        raise ValueError("Saved pipeline predictions differ after reload")
    return {"sha256": sha256(path), "round_trip_verified": True, "score_kind": score_kind,
            f"max_{score_kind}_difference": float(np.max(np.abs(original - reloaded)))}


def source_files(repository=None) -> list[str]:
    """Include all local source dependencies so refactors remain reproducible."""
    repository = Path(repository) if repository is not None else repo_path(".")
    names = [path.relative_to(repository).as_posix()
             for folder in ("src", "configs", "tests")
             for path in (repository / folder).rglob("*")
             if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md"}]
    names += [name for name in ("requirements.txt", "requirements-lock.txt", "pytest.ini")
              if (repository / name).is_file()]
    return sorted(names)


def snapshot_sources(names: list[str], root, *, path_resolver=None) -> dict:
    """Preserve run code/config alongside hashes, including uncommitted files."""
    resolve = path_resolver or repo_path
    hashes = {}
    for name in names:
        source = resolve(name)
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[name] = sha256(source)
        if sha256(destination) != hashes[name]:
            raise ValueError(f"Source changed while recording provenance: {name}")
    return hashes
