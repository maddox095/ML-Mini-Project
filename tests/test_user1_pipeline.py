from __future__ import annotations

import pandas as pd

from src.data.build_dataset import build_dataset


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
