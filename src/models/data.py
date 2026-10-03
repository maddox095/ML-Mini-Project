"""Load a hash-verified partition from the current User 1 handoff."""

import json

import pandas as pd

from src.data.common import model_features, repo_path, sha256

PROTOCOLS = ("stratified_partition", "artist_disjoint_partition")


def load_partition(protocol: str, partition: str, *, allow_test: bool = False):
    """Return inputs, target and metadata; test access is explicit."""
    if protocol not in PROTOCOLS or partition not in {"train", "test"}:
        raise ValueError(f"Unknown partition: {protocol}/{partition}")
    if partition == "test" and not allow_test:
        raise ValueError("Test partition is reserved for final evaluation; request allow_test=True only after model selection.")
    root = repo_path("data/processed")
    manifest = json.loads((root / "user2_split_manifest.json").read_text(encoding="utf-8"))
    source = repo_path(manifest["source"])
    split_path = root / "user2_splits.csv"
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Model table has changed; regenerate splits before use")
    if sha256(split_path) != manifest.get("splits_sha256"):
        raise ValueError("Split assignments have changed; regenerate splits before use")
    if manifest["features"] != model_features() or manifest["target"] != "hit":
        raise ValueError("Split manifest does not match the configured input whitelist")
    table = pd.read_parquet(source)
    splits = pd.read_csv(split_path)
    if (not table.track_id.is_unique or not splits.track_id.is_unique
            or set(table.track_id) != set(splits.track_id)):
        raise ValueError("Split identities do not match the model table")
    data = table.merge(splits[["track_id", protocol]], on="track_id", validate="one_to_one")
    selected = data.loc[data[protocol].eq(partition)].copy()
    if selected.empty:
        raise ValueError(f"Empty partition: {protocol}/{partition}")
    return selected[manifest["features"]], selected[manifest["target"]], selected
