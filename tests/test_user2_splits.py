import pandas as pd
import pytest

from src.data.build_splits import build_splits


def fixture_table():
    return pd.DataFrame({
        "track_id": [f"t{i:02}" for i in range(40)],
        "canonical_key": [f"song{i}" for i in range(40)],
        "artist_key": [f"artist{i // 2}" for i in range(40)],
        "hit": [i % 2 for i in range(40)],
    })


def test_splits_are_order_independent_complete_and_disjoint():
    table = fixture_table()
    splits = build_splits(table)
    pd.testing.assert_frame_equal(splits, build_splits(table.sample(frac=1, random_state=7)))
    assert set(splits.track_id) == set(table.track_id)
    assert splits.track_id.is_unique
    test = splits[splits.stratified_partition.eq("test")]
    assert test.hit.value_counts().to_dict() == {0: 5, 1: 5}
    train_artists = set(splits.loc[splits.artist_disjoint_partition.eq("train"), "artist_key"])
    test_artists = set(splits.loc[splits.artist_disjoint_partition.eq("test"), "artist_key"])
    assert train_artists.isdisjoint(test_artists)


def test_duplicate_identities_rejected():
    table = fixture_table()
    table.loc[1, "canonical_key"] = table.loc[0, "canonical_key"]
    with pytest.raises(ValueError, match="unique"):
        build_splits(table)
