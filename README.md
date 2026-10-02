# HitPredict - User 1 data pipeline

This repository implements the **User 1** half of the HitPredict reproduction:
immutable input registration, Million Song Dataset (MSD) extraction, Billboard
history loading, conservative identity matching, overlap removal, target labels,
chronological Artist Score, QA/EDA, and a deterministic handoff table for User 2.

The supplied guide makes one constraint explicit: do not newly collect Spotify
audio features for ML training. This implementation therefore defaults to **Mode
B**, a current-compliant methodology reproduction using MSD-native `tempo`,
`loudness`, and `duration`. It does not claim feature-for-feature or numerical
reproduction of the 2018 Spotify-based experiment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Build the User 1 handoff

Raw input files remain local and are excluded from Git. Each acquisition command
records source URL, retrieval time, size, SHA-256, and rows in
`data/raw/MANIFEST.csv`.

```bash
# 1. Download the full MSD summary HDF5 from the URL in configs/data.yaml,
#    then save it as data/raw/msd_full/msd_summary_file.h5.
# 2. Download and normalize the historical Billboard weekly archive.
python -m src.data.load_billboard --download

# 3. Extract the full MSD summary and construct the audited handoff.
python -m src.data.extract_msd
python -m src.data.build_dataset
pytest -q
```

The pipeline creates `data/interim/model_table.parquet` (4,000 balanced rows)
plus
`reports/match_audit.csv`, `reports/artist_score_audit.csv`,
`reports/data_quality.csv`, `reports/feature_availability.csv`, and
`reports/eda_summary.md`.

The workflow uses the official million-track MSD summary file, a 1990-2018
label window, and 1986-2018 Billboard history only for Artist Score. The raw
summary HDF5 remains excluded from Git; its source and SHA-256 are recorded in
`data/raw/MANIFEST.csv`.

The model table contains the valid MSD summary descriptors `tempo`,
`loudness`, and `duration`, plus chronological `artist_score`. The source
file's `danceability` and `energy` columns are constant zero in this release,
so they are explicitly excluded rather than treated as model inputs.

## Data rules that are enforced

- Original artist/title strings are retained; normalized keys are comparison-only.
- The pipeline accepts only exact canonical artist/title identities; unresolved labels
  remain visible in the matching audit rather than being guessed.
- MSD candidates that match any Billboard hit are removed before sampling the
  non-hit pool. Balancing happens only after this cleanup and uses seed 42.
- `artist_score=1` only when the audit identifies an earlier chart event for that
  same normalized artist. The job fails if chronology is invalid.
- Chart rank, peak position, and weeks on chart are kept in raw audit data only;
  they are never model inputs.

User 2 should consume only the accepted `model_table.parquet`, fit preprocessing
inside training pipelines, and keep the prescribed 75/25 source-comparison split
separate from a true held-out evaluation.
