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
# 1. Obtain and register the MSD subset (large download; do this intentionally).
python -m src.data.download_msd --download
# Or register a file already downloaded elsewhere:
python -m src.data.download_msd --archive /absolute/path/msd_subset.tar.gz

# Extract the archive yourself, then make the canonical MSD metadata table.
python -m src.data.extract_msd --input-root data/raw/msd/extracted

# 2. Download and normalize the historical Billboard weekly archive.
python -m src.data.load_billboard --download

# 3. Construct the deterministic, audited User 1 handoff.
python -m src.data.build_dataset
```

The final command creates `data/interim/model_table.parquet` plus
`reports/match_audit.csv`, `reports/artist_score_audit.csv`,
`reports/data_quality.csv`, `reports/feature_availability.csv`, and
`reports/eda_summary.md`.

## Data rules that are enforced

- Original artist/title strings are retained; normalized keys are comparison-only.
- Matching is exact first. Fuzzy candidates are conservative and auditable; they
  are never silently converted into labels.
- MSD candidates that match any Billboard hit are removed before sampling the
  non-hit pool. Balancing happens only after this cleanup and uses seed 42.
- `artist_score=1` only when the audit identifies an earlier chart event for that
  same normalized artist. The job fails if chronology is invalid.
- Chart rank, peak position, and weeks on chart are kept in raw audit data only;
  they are never model inputs.

User 2 should consume only the accepted `model_table.parquet`, fit preprocessing
inside training pipelines, and keep the prescribed 75/25 source-comparison split
separate from a true held-out evaluation.
