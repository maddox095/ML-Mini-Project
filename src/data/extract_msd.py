"""Extract the required fields from the official MSD million-track summary HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd
from tqdm import trange

from src.data.common import append_manifest, repo_path
from src.features.normalize_names import canonicalize_frame


def _decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def extract_summary(input_path: Path, chunk_size: int = 100_000) -> pd.DataFrame:
    """Return one row per MSD summary-file track with Mode B descriptors."""
    chunks: list[pd.DataFrame] = []
    with h5py.File(input_path, "r") as handle:
        metadata = handle["metadata/songs"]
        analysis = handle["analysis/songs"]
        musicbrainz = handle["musicbrainz/songs"]
        if not (len(metadata) == len(analysis) == len(musicbrainz)):
            raise ValueError("MSD summary tables have inconsistent row counts")
        for start in trange(0, len(metadata), chunk_size, desc="Extracting MSD summary"):
            stop = min(start + chunk_size, len(metadata))
            meta = metadata[start:stop]
            audio = analysis[start:stop]
            years = musicbrainz[start:stop]
            chunks.append(
                pd.DataFrame(
                    {
                        "track_id": [_decode(value) for value in audio["track_id"]],
                        "song_id": [_decode(value) for value in meta["song_id"]],
                        "title": [_decode(value) for value in meta["title"]],
                        "artist_name": [_decode(value) for value in meta["artist_name"]],
                        "year": years["year"],
                        "duration": audio["duration"],
                        "tempo": audio["tempo"],
                        "loudness": audio["loudness"],
                    }
                )
            )
    return canonicalize_frame(pd.concat(chunks, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/msd_full/msd_summary_file.h5")
    parser.add_argument("--output", default="data/interim/msd_summary_tracks.parquet")
    parser.add_argument("--source-url", default="http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5")
    args = parser.parse_args()
    input_path, output_path = repo_path(args.input), repo_path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    tracks = extract_summary(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(output_path, index=False)
    append_manifest(input_path, source="Million Song Dataset summary file", source_url=args.source_url, row_count=len(tracks))
    quality = {
        "rows": len(tracks),
        "year_zero_or_missing": int(tracks.year.isna().sum() + tracks.year.eq(0).sum()),
        "duplicate_canonical_rows": int(tracks.duplicated("canonical_key", keep=False).sum()),
        "year_min": int(tracks.year.min()),
        "year_max": int(tracks.year.max()),
    }
    report = repo_path("reports/msd_summary_quality.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tracks):,} MSD summary tracks to {output_path}")


if __name__ == "__main__":
    main()
