# Immutable raw data

Keep downloaded archives and source exports here, without editing them.  Register
each input with `python -m src.data.download_msd --archive ...` or
`python -m src.data.load_billboard --download`; both commands append source,
size, SHA-256, date and row-count information to `data/raw/MANIFEST.csv`.

The default project mode is a methodology reproduction using Million Song Dataset
features. It intentionally does not collect Spotify audio features for ML
training. Do not commit large or restricted raw data files.
