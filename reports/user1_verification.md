# User 1 verification

- Dataset version: `u1-mode-b`.
- Rows: 4,000; hits: 2,000; non-hits: 2,000.
- Integrity checks: 27/27 passed.
- Status: handoff integrity passed; source review required.
- Dataset SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.

This report is generated from the current table and audits. Regenerate it with
`python -m src.data.verify_handoff` after data changes.

## Checks

| Check | Result |
| --- | --- |
| required_columns | PASS |
| nonempty | PASS |
| one_row_per_identity | PASS |
| unique_track_ids | PASS |
| no_conflicting_labels | PASS |
| binary_target_both_classes | PASS |
| balanced_classes | PASS |
| binary_artist_score | PASS |
| required_fields_present | PASS |
| nonempty_identities | PASS |
| finite_features | PASS |
| positive_tempo_and_duration | PASS |
| dates_in_label_window | PASS |
| history_contains_label_window | PASS |
| model_features_exclude_outcomes | PASS |
| handoff_excludes_matching_provenance | PASS |
| artist_audit_schema | PASS |
| artist_audit_covers_all_rows | PASS |
| artist_audit_scores_agree | PASS |
| artist_audit_names_agree | PASS |
| artist_audit_dates_agree | PASS |
| positive_artist_scores_have_prior_evidence | PASS |
| prior_evidence_in_history_window | PASS |
| zero_artist_scores_have_no_prior_evidence | PASS |
| chronology_flags_pass | PASS |
| feature_sources_documented | PASS |
| quality_counts_agree | PASS |

## Source availability

| Artifact | Present | Hash and size verified |
| --- | --- | --- |
| `data/raw/billboard/all.json` | False | False |
| `data/raw/msd_full/msd_summary_file.h5` | False | False |

## Remaining limitations

- Registered raw sources are unavailable or unverified locally; full-source extraction and labels cannot be independently reproduced in this checkout.
- Extracted MSD/Billboard intermediate tables are absent locally.
- Mode B uses three MSD audio descriptors plus Artist Score; it is not a numerical reproduction of the original Spotify-feature experiment.
- Exact matching and finite chart history can leave label noise; an unmatched song is only a non-hit candidate.
- Non-hit reference dates approximate release as January 1; dates are metadata, not model inputs.
- Saved Artist Score evidence validates chronology, not every source event or zero score without raw history.
