# Immutable raw data

Keep downloaded archives and source exports here, without editing them. Save the
official MSD summary file as `msd_full/msd_summary_file.h5`, then run
`python -m src.data.extract_msd`. Use `python -m src.data.load_billboard
--download` for Billboard data. Both pipeline commands register source, size,
SHA-256, date and row-count information in `data/raw/MANIFEST.csv`.

The default project mode is a methodology reproduction using Million Song Dataset
features. It intentionally does not collect Spotify audio features for ML
training. Do not commit large or restricted raw data files.
