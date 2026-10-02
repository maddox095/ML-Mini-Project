from __future__ import annotations

import pandas as pd

from src.data.build_dataset import build_handoff
from src.features.normalize_names import normalize_key


def config() -> dict:
    return {"seed": 42, "date_window": {"start": "1990-01-01", "end": "2018-12-31"}, "matching": {"fuzzy_enabled": False, "strip_title_versions": False}, "balancing": {"enabled": False}, "mode_b_features": ["tempo", "loudness", "duration"], "reproduction_mode": "B"}


def test_normalize_key_preserves_comparable_identity() -> None:
    assert normalize_key("Beyoncé — Halo!") == "beyonce halo"


def test_handoff_removes_hit_from_negative_pool_and_uses_prior_history() -> None:
    tracks = pd.DataFrame([
        {"track_id": "A", "song_id": "A", "title": "First Hit", "artist_name": "Artist A", "year": 2000, "tempo": 100, "loudness": -8, "duration": 200},
        {"track_id": "B", "song_id": "B", "title": "Later Hit", "artist_name": "Artist A", "year": 2003, "tempo": 110, "loudness": -7, "duration": 210},
        {"track_id": "C", "song_id": "C", "title": "Non Hit", "artist_name": "Artist A", "year": 2004, "tempo": 120, "loudness": -9, "duration": 220},
    ])
    charts = pd.DataFrame([
        {"chart_date": "2000-06-01", "title": "First Hit", "artist_name": "Artist A", "rank": 1},
        {"chart_date": "2003-06-01", "title": "Later Hit", "artist_name": "Artist A", "rank": 2},
    ])
    result = build_handoff(tracks, charts, config())
    table = result.model_table.set_index("track_id")
    assert set(table.index) == {"A", "B", "C"}
    assert table.loc["A", "artist_score"] == 0
    assert table.loc["B", "artist_score"] == 1
    assert table.loc["C", "artist_score"] == 1
    assert result.artist_score_audit["chronology_valid"].all()
    assert table.groupby("canonical_key")["hit"].nunique().max() == 1
