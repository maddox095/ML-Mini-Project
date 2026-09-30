"""Download and normalize a historical Billboard archive into a raw-derived table."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.common import append_manifest, load_config, repo_path
from src.features.normalize_names import canonicalize_frame


def _items(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("charts", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unrecognised Billboard archive: expected a list of weekly charts")


def _value(entry: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in entry:
            return entry[name]
    return None


def parse_billboard(payload: Any, start: str, end: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lower, upper = pd.Timestamp(start), pd.Timestamp(end)
    for chart in _items(payload):
        chart_date = pd.to_datetime(_value(chart, "date", "chart_date"), errors="coerce")
        if pd.isna(chart_date) or not (lower <= chart_date <= upper):
            continue
        entries = _value(chart, "data", "entries", "songs") or []
        for entry in entries:
            title = _value(entry, "song", "title", "name")
            artist = _value(entry, "artist", "artist_name", "performer")
            if not title or not artist:
                continue
            rows.append(
                {
                    "chart_date": chart_date,
                    "title": str(title),
                    "artist_name": str(artist),
                    "rank": _value(entry, "this_week", "rank", "position"),
                    "peak_position": _value(entry, "peak_position", "peak"),
                    "weeks_on_chart": _value(entry, "weeks_on_chart", "weeks"),
                }
            )
    if not rows:
        raise ValueError("No Billboard rows remained in the configured date window")
    return canonicalize_frame(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Existing all.json archive")
    parser.add_argument("--download", action="store_true", help="Fetch configured historical archive")
    args = parser.parse_args()
    if bool(args.input) == bool(args.download):
        parser.error("supply exactly one of --input or --download")
    config = load_config()
    url = config["sources"]["billboard_archive_url"]
    raw_path = repo_path(args.input) if args.input else repo_path("data/raw/billboard/all.json")
    if args.download:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable raw file: {raw_path}")
        with urllib.request.urlopen(url) as response, raw_path.open("xb") as output:
            output.write(response.read())
    with raw_path.open(encoding="utf-8") as handle:
        billboard = parse_billboard(json.load(handle), config["date_window"]["start"], config["date_window"]["end"])
    output = repo_path("data/interim/billboard_entries.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    billboard.to_parquet(output, index=False)
    append_manifest(raw_path, source="Billboard Hot 100 historical archive", source_url=url, row_count=len(billboard))
    print(f"Wrote {len(billboard):,} weekly chart rows to {output}")


if __name__ == "__main__":
    main()
