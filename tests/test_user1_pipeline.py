from __future__ import annotations

import pandas as pd
import pytest
import json
from unittest.mock import MagicMock, patch

from src.data.build_dataset import build_dataset
from src.features.normalize_names import base_title, normalize_key
from src.data.load_billboard import download_billboard


def test_uses_pre_label_history_for_artist_score() -> None:
    config = {
        "seed": 42,
        "label_window": {"start": "1990-01-01", "end": "2018-12-31"},
        "artist_history_window": {"start": "1986-01-01", "end": "2018-12-31"},
        "matching": {"strip_title_versions": False},
        "balancing": {"target_rows_per_class": 1},
    }
    msd = pd.DataFrame(
        [
            {"track_id": "hit", "song_id": "hit", "title": "Later Hit", "artist_name": "Artist A", "year": 1991, "duration": 200, "tempo": 110, "loudness": -7, "danceability": 0.5, "energy": 0.6},
            {"track_id": "nonhit", "song_id": "nonhit", "title": "Non Hit", "artist_name": "Artist B", "year": 1992, "duration": 210, "tempo": 120, "loudness": -8, "danceability": 0.4, "energy": 0.5},
        ]
    )
    billboard = pd.DataFrame(
        [
            {"chart_date": "1987-06-01", "title": "Early Hit", "artist_name": "Artist A", "rank": 1},
            {"chart_date": "1991-06-01", "title": "Later Hit", "artist_name": "Artist A", "rank": 1},
        ]
    )
    final, _, audit, quality = build_dataset(msd, billboard, config)
    hit = final.loc[final.hit.eq(1)].iloc[0]
    assert hit.artist_score == 1
    assert audit.chronology_valid.all()
    assert int(quality.loc[quality.metric.eq("final_rows"), "value"].iloc[0]) == 2
    assert not {"source", "match_method", "match_score"}.intersection(final.columns)


@pytest.mark.parametrize("title", ["Alive", "Live", "Long Live", "Deliver"])
def test_title_normalization_preserves_ordinary_titles(title: str) -> None:
    assert base_title(title, strip_versions=True) == normalize_key(title)


@pytest.mark.parametrize("title", ["Song (Live)", "Song - Live", "Song [Remastered 2010]"])
def test_title_normalization_strips_delimited_versions(title: str) -> None:
    assert base_title(title, strip_versions=True) == "song"


def test_missing_identity_is_not_a_literal_string() -> None:
    assert normalize_key(pd.NA) == ""
    assert base_title(float("nan")) == ""


def test_historical_hits_and_invalid_features_are_excluded_from_negatives() -> None:
    config = {
        "seed": 42,
        "label_window": {"start": "1990-01-01", "end": "2018-12-31"},
        "artist_history_window": {"start": "1986-01-01", "end": "2018-12-31"},
        "matching": {"strip_title_versions": False},
        "balancing": {"target_rows_per_class": 10},
    }
    msd = pd.DataFrame([
        {"track_id": title, "song_id": title, "title": title, "artist_name": "Artist", "year": 1991,
         "duration": 200, "tempo": tempo, "loudness": -7}
        for title, tempo in [("Reentry", 100), ("Old Hit", 100), ("Non Hit", 100), ("Unknown Tempo", 0)]
    ])
    history = pd.DataFrame([
        {"chart_date": date, "title": title, "artist_name": "Artist", "rank": 1}
        for date, title in [("1987-01-01", "Reentry"), ("1988-01-01", "Old Hit"), ("1991-06-01", "Reentry")]
    ])
    final, _, audit, _ = build_dataset(msd, history, config)
    assert final.loc[final.hit.eq(0), "title"].tolist() == ["Non Hit"]
    assert audit.loc[audit.canonical_key.eq("artist || reentry"), "prior_hit_title"].iloc[0] == "Old Hit"
    with pytest.raises(ValueError, match="both positive and negative"):
        build_dataset(msd.loc[msd.title.eq("Non Hit")], history, config)


def test_interrupted_download_can_be_retried(tmp_path) -> None:
    raw_path = tmp_path / "all.json"
    response = MagicMock()
    response.__enter__.return_value.read.side_effect = [b"partial", ConnectionResetError("interrupted")]
    with patch("src.data.load_billboard.urllib.request.urlopen", return_value=response):
        with pytest.raises(ConnectionResetError):
            download_billboard("https://example.test/archive", raw_path, "1990-01-01", "1991-12-31")
    assert list(tmp_path.iterdir()) == []
    payload = json.dumps([{"date": "1991-01-01", "data": [{"song": "Song", "artist": "Artist", "rank": 1}]}]).encode()
    response.__enter__.return_value.read.side_effect = [payload, b""]
    with patch("src.data.load_billboard.urllib.request.urlopen", return_value=response):
        download_billboard("https://example.test/archive", raw_path, "1990-01-01", "1991-12-31")
        with pytest.raises(FileExistsError):
            download_billboard("https://example.test/archive", raw_path, "1990-01-01", "1991-12-31")
    assert raw_path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [raw_path]


def test_invalid_download_is_not_published(tmp_path) -> None:
    response = MagicMock()
    response.__enter__.return_value.read.side_effect = [b"invalid json", b""]
    with patch("src.data.load_billboard.urllib.request.urlopen", return_value=response):
        with pytest.raises(json.JSONDecodeError):
            download_billboard("https://example.test/archive", tmp_path / "all.json", "1990-01-01", "1991-12-31")
    assert list(tmp_path.iterdir()) == []
