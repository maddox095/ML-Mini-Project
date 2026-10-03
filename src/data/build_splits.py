"""Create reproducible User 2 partitions without model preprocessing."""
import json

import numpy as np
import pandas as pd

from src.data.common import load_config, model_features, repo_path, sha256


def build_splits(table: pd.DataFrame, *, seed: int | None = None) -> pd.DataFrame:
    if not table.track_id.is_unique or not table.canonical_key.is_unique:
        raise ValueError("Split identities must be unique")
    if table.artist_key.isna().any() or set(table.hit.unique()) != {0, 1}:
        raise ValueError("Require artist keys and both binary classes")
    table = table.sort_values("track_id").reset_index(drop=True)
    rng = np.random.default_rng(load_config()["seed"] if seed is None else seed)
    result = table[["track_id", "canonical_key", "artist_key", "hit"]].copy()
    result["stratified_partition"] = "train"
    for label in (0, 1):
        indices = rng.permutation(table.index[table.hit.eq(label)])
        n_test = round(len(indices) * 0.25)
        result.loc[indices[:n_test], "stratified_partition"] = "test"
    # Select whole artists; the resulting row/class proportions are approximate.
    artists = rng.permutation(sorted(table.artist_key.unique()))
    test_artists = set(artists[:round(len(artists) * 0.25)])
    result["artist_disjoint_partition"] = np.where(
        table.artist_key.isin(test_artists), "test", "train"
    )
    for column in ("stratified_partition", "artist_disjoint_partition"):
        for partition in ("train", "test"):
            if set(result.loc[result[column].eq(partition), "hit"]) != {0, 1}:
                raise ValueError(f"Both classes required in {column}/{partition}")
    return result


def main() -> None:
    config = load_config()
    source = repo_path("data/interim/model_table.parquet")
    table = pd.read_parquet(source)
    splits = build_splits(table, seed=config["seed"])
    output = repo_path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output / "user2_splits.csv", index=False, lineterminator="\n")
    summary = {}
    for column in ("stratified_partition", "artist_disjoint_partition"):
        summary[column] = {
            partition: {
                "rows": int(group.shape[0]),
                "hits": int(group.hit.sum()),
                "non_hits": int(group.hit.eq(0).sum()),
                "artists": int(group.artist_key.nunique()),
            }
            for partition, group in splits.groupby(column)
        }
        train = set(splits.loc[splits[column].eq("train"), "artist_key"])
        test = set(splits.loc[splits[column].eq("test"), "artist_key"])
        summary[column]["shared_artists"] = len(train & test)
    manifest = {
        "source": source.relative_to(repo_path(".")).as_posix(),
        "source_sha256": sha256(source),
        "splits_sha256": sha256(output / "user2_splits.csv"),
        "dataset_version": config["dataset_version"],
        "seed": config["seed"],
        "features": model_features(config),
        "target": "hit",
        "splits": summary,
    }
    (output / "user2_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
