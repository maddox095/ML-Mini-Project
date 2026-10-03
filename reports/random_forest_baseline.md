# Random Forest baseline

Run: `random_forest_baseline_v1`. Device: CPU. Status: preliminary Mode B experiment.

Scores below are mean +/- sample standard deviation across 5 outer training-validation folds.
Each Random Forest fold tunes its parameters in 5 separate inner folds. The final test sets were not evaluated.

| Protocol | Model | Features | CV accuracy | CV ROC-AUC | CV F1 | Selected C |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| stratified_partition | dummy | prior | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.000 | - |
| stratified_partition | random_forest | audio_only | 0.671 +/- 0.018 | 0.720 +/- 0.012 | 0.722 | - |
| stratified_partition | random_forest | audio_artist_score | 0.827 +/- 0.017 | 0.886 +/- 0.013 | 0.806 | - |
| stratified_partition | random_forest | artist_score_only | 0.824 +/- 0.017 | 0.824 +/- 0.017 | 0.802 | - |
| artist_disjoint_partition | dummy | prior | 0.501 +/- 0.000 | 0.500 +/- 0.000 | 0.668 | - |
| artist_disjoint_partition | random_forest | audio_only | 0.671 +/- 0.019 | 0.723 +/- 0.014 | 0.717 | - |
| artist_disjoint_partition | random_forest | audio_artist_score | 0.824 +/- 0.012 | 0.884 +/- 0.011 | 0.802 | - |
| artist_disjoint_partition | random_forest | artist_score_only | 0.823 +/- 0.011 | 0.824 +/- 0.011 | 0.800 | - |

## Candidate for later comparison

The current Random Forest candidate uses `audio_artist_score` under the artist-disjoint protocol.
Pipeline: `models/random_forest_baseline_v1/artist_disjoint_partition__random_forest__audio_artist_score.joblib`. This is a baseline candidate, not the final model for deployment.
Final refit parameters are selected by CV on all outer training rows; outer folds tune independently.

## Training-only diagnostics

- fixed_parameters_seed_stability: mean accuracy 0.825; mean ROC-AUC 0.884, SD across seed means 0.0035.
- shuffled_training_labels: mean accuracy 0.504; mean ROC-AUC 0.506, SD across seed means 0.0361.

All 8 saved pipelines passed save/reload score and class agreement checks.
Audio-only and Artist-Score-only comparisons quantify the influence of artist history.
These are training-validation results; CV-guided model/feature selection still requires final held-out evaluation.

## Provenance and limitations

- Dataset SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.
- Split SHA-256: `bccaf24876a5914e2afc03ce3388fcdd0cc8fa6052f9ea52f6c5d38964e29a20`.
- Full metrics, fold assignments, searches, coefficients, predictions and manifest: `reports/runs/random_forest_baseline_v1`.
- Feature coefficients/importances, where available, are associations rather than causal effects.
- Raw sources remain absent locally; source-label reproducibility and exact numerical reproduction are not established.
- Balanced sampling probabilities do not estimate commercial success probabilities for natural music releases.

Reproduce with `python -m src.models.train --model random_forest`; each run creates separate artifact directories.
