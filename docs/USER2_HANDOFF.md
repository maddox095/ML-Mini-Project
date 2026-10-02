# User 2 modeling handoff

Use `data/interim/model_table.parquet`: 4,000 songs, with 2,000 hits and
2,000 non-hit candidates. The four permitted inputs have no missing values.

| Role | Columns |
| --- | --- |
| Audio inputs | `tempo` (BPM), `loudness` (dB), `duration` (seconds) |
| Historical input | `artist_score` (0/1: prior hit by the same normalized artist) |
| Target | `hit` (0/1) |
| Metadata only | All other columns, including IDs, names, keys, year and dates |

## Reproduce the partitions

From the repository root, after building the dataset:

```bash
python -m src.data.build_splits
```

`data/processed/user2_splits.csv` assigns every track to two independent
experiments. `user2_split_manifest.json` records seed 42, the source file
SHA-256, features, counts and artist overlap. Regenerate these files after any
change to the model table; verify the source hash before using existing splits.

| Experiment | Train | Test | Shared artists |
| --- | ---: | ---: | ---: |
| Stratified 75/25 | 3,000 (1,500 hits) | 1,000 (500 hits) | 263 |
| Artist-disjoint | 3,024 (1,516 hits) | 976 (484 hits) | 0 |

The stratified partition is a conventional held-out baseline. It does not
implement or verify the original study's prescribed source-comparison protocol;
that protocol remains a separate experiment to confirm against the project
guide. The PDFs in this checkout are Git LFS pointers, so their contents were
not available when preparing this handoff.

The artist-disjoint partition randomly reserves 25% of unique normalized artist
keys, so its row and class proportions are approximate. Different credits or
collaborations can represent the same performer under different keys. This is
an evaluation on unseen artist keys, not a future-release evaluation.

## Load the data

```python
import hashlib
import json
from pathlib import Path
import pandas as pd

source = Path("data/interim/model_table.parquet")
manifest = json.loads(Path("data/processed/user2_split_manifest.json").read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
table = pd.read_parquet(source)
splits = pd.read_csv("data/processed/user2_splits.csv")
assert set(table.track_id) == set(splits.track_id)
data = table.merge(
    splits[["track_id", "stratified_partition", "artist_disjoint_partition"]],
    on="track_id", validate="one_to_one",
)
partition = "stratified_partition"  # Repeat independently with artist_disjoint_partition.
features = ["tempo", "loudness", "duration", "artist_score"]
train = data.loc[data[partition].eq("train")]
test = data.loc[data[partition].eq("test")]
X_train, y_train = train[features], train["hit"]
X_test, y_test = test[features], test["hit"]
```

Fit scaling, imputation if needed, feature selection and tuning using training
rows only. Put preprocessing inside the model pipeline. Use cross-validation
within training data for tuning; use artist groups for the artist-disjoint
experiment's validation folds. Keep each experiment's test rows out of its own
tuning process, and freeze the approach before evaluating either test set.

## Suggested experiment checklist

- Compare a majority-class baseline and your chosen classifiers.
- Evaluate audio-only inputs against audio plus `artist_score`, using identical
  partitions. Artist Score's descriptive correlation with `hit` is 0.662;
  the ablation measures how much it contributes to prediction.
- Report accuracy, precision, recall, F1, ROC-AUC and a confusion matrix. Include
  the split name, seed, feature set and test class counts with each result.
- Review duration outliers (the maximum non-hit duration is 2,513.16 seconds).
  Any clipping thresholds must be chosen on training data only.

## Interpretation and limitations

This is a Mode B methodology reproduction with MSD features, not a numerical
reproduction of the original Spotify feature experiment. `danceability` and
`energy` are excluded because they are constant zero in this MSD release.

Only 2,538 of 10,878 unique Billboard songs in the label window matched MSD
features before sampling; 8,340 remained unresolved. Exact matching and finite
chart history can leave label noise. A non-hit candidate is not proof that a
song never charted. Balanced sampling also means test precision and predicted
probabilities should not be presented as estimates for naturally occurring
music releases without further validation.

Artist Score uses strictly earlier chart events and excludes the current song.
Positive reference dates are first chart appearances within the label window;
negative dates approximate release as January 1 of the MSD year. These dates
must remain metadata. Earlier chart history remains available for unseen artist
keys: artist-disjoint evaluation does not mean predicting without artist history.

Audit details are in `reports/data_quality.csv`, `reports/match_audit.csv`,
`reports/artist_score_audit.csv`, `reports/feature_availability.csv` and
`reports/eda_summary.md`.
