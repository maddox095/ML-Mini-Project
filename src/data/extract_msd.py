"""Extract the User 1 MSD metadata contract from an existing HDF5 tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from tqdm import tqdm

from src.data.common import repo_path
from src.features.normalize_names import canonicalize_frame

FIELDS = ("track_id", "song_id", "title", "artist_name", "year", "duration", "tempo", "loudness")


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.item() if hasattr(value, "item") else value


def _read_field(handle: h5py.File, field: str) -> Any:
    found: list[Any] = []
    def visitor(_: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.dtype.names and field in obj.dtype.names and len(obj):
            found.append(obj[0][field])
    handle.visititems(visitor)
    return _decode(found[0]) if found else None


def extract_file(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        row = {field: _read_field(handle, field) for field in FIELDS}
    row["source_h5"] = str(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Extracted MSD directory")
    parser.add_argument("--output", default="data/interim/msd_tracks.parquet")
    args = parser.parse_args()
    input_root = repo_path(args.input_root)
    files = sorted(input_root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {input_root}")
    rows = [extract_file(path) for path in tqdm(files, desc="Extracting MSD HDF5")]
    tracks = canonicalize_frame(pd.DataFrame(rows))
    for column in ("year", "duration", "tempo", "loudness"):
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(output, index=False)
    quality = {
        "h5_files": len(files),
        "rows": len(tracks),
        "year_zero_or_missing": int(tracks["year"].isna().sum() + tracks["year"].eq(0).sum()),
        "duplicate_canonical_keys": int(tracks.duplicated("canonical_key", keep=False).sum()),
        "year_min": None if tracks["year"].dropna().empty else int(tracks["year"].min()),
        "year_max": None if tracks["year"].dropna().empty else int(tracks["year"].max()),
        "missing_by_field": tracks[list(FIELDS)].isna().sum().to_dict(),
    }
    report = repo_path("reports/msd_extraction_quality.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(quality, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {len(tracks):,} MSD tracks to {output}")


if __name__ == "__main__":
    main()
