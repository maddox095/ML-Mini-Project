# Reproduction deviation log

| Source says | Reproduction does | Reason |
|---|---|---|
| Use Spotify audio descriptors for the original nine audio features. | Defaults to Mode B with MSD-native tempo, loudness and duration. | Current Spotify policy/access constraints make new audio-feature collection unsuitable for ML training. |
| Download the MSD 10k subset from the historical official host. | Accepts a locally acquired MSD archive through `download_msd --archive` and records its hash. | On 2026-10-01, the legacy direct URL in the initial config returned HTTP 404. The guide's Academic Torrents mirror remains the recommended acquisition route. |
| 1990-2018 collection is mentioned alongside an inconsistent 1991-2010 label sentence. | Uses 1990-01-01 through 2018-12-31 consistently. | This is the guide's stated default interpretation and is explicit in `configs/data.yaml`. |
| Artist Score indicates a previous Billboard hit. | Requires a saved strictly earlier first-chart event for every positive Artist Score. | Avoids future-career leakage. |
