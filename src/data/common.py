"""Small shared helpers for deterministic, auditable data jobs."""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    with repo_path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_features(config: dict[str, Any] | None = None) -> list[str]:
    """Return the input whitelist shared by data checks and model pipelines."""
    if config is None:
        config = load_config()
    return [*config["mode_b_features"], "artist_score"]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def append_manifest(path: Path, *, source: str, source_url: str, row_count: int | str = "") -> None:
    """Register an input once per immutable artifact checksum."""
    manifest = ROOT / "data/raw/MANIFEST.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists()
    record = {
        "artifact": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "source": source,
        "source_url": source_url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "row_count": row_count,
    }
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as handle:
            for existing in csv.DictReader(handle):
                if existing.get("artifact") == record["artifact"] and existing.get("sha256") == record["sha256"]:
                    return
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)
