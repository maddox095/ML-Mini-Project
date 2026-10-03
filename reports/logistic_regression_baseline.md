# Logistic Regression baseline

Run: `logistic_regression_baseline_v2`. Device: CPU. Status: preliminary Mode B experiment.

Scores below are mean +/- sample standard deviation across 5 outer training-validation folds.
Each Logistic Regression fold tunes its parameters in 5 separate inner folds. The final test sets were not evaluated.

| Protocol | Model | Features | CV accuracy | CV ROC-AUC | CV F1 | Selected C |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| stratified_partition | dummy | prior | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.000 | - |
| stratified_partition | logistic_regression | audio_only | 0.619 +/- 0.018 | 0.664 +/- 0.020 | 0.644 | 1 |
| stratified_partition | logistic_regression | audio_artist_score | 0.822 +/- 0.011 | 0.873 +/- 0.015 | 0.802 | 0.01 |
| stratified_partition | logistic_regression | artist_score_only | 0.824 +/- 0.017 | 0.824 +/- 0.017 | 0.802 | 0.01 |
| artist_disjoint_partition | dummy | prior | 0.501 +/- 0.000 | 0.500 +/- 0.000 | 0.668 | - |
| artist_disjoint_partition | logistic_regression | audio_only | 0.619 +/- 0.025 | 0.666 +/- 0.023 | 0.647 | 1 |
| artist_disjoint_partition | logistic_regression | audio_artist_score | 0.820 +/- 0.011 | 0.870 +/- 0.015 | 0.798 | 100 |
| artist_disjoint_partition | logistic_regression | artist_score_only | 0.823 +/- 0.011 | 0.824 +/- 0.011 | 0.800 | 0.01 |

## Candidate for later comparison

The current Logistic Regression candidate uses `audio_artist_score` under the artist-disjoint protocol.
Pipeline: `models/logistic_regression_baseline_v2/artist_disjoint_partition__logistic_regression__audio_artist_score.joblib`. This is a baseline candidate, not the final model for deployment.
Final refit parameters are selected by CV on all outer training rows; outer folds tune independently.

## Training-only diagnostics

- fixed_parameters_seed_stability: mean accuracy 0.822; mean ROC-AUC 0.871, SD across seed means 0.0003.
- shuffled_training_labels: mean accuracy 0.525; mean ROC-AUC 0.524, SD across seed means 0.1254.

All 8 saved pipelines passed save/reload score and class agreement checks.
Audio-only and Artist-Score-only comparisons quantify the influence of artist history.
These are training-validation results; CV-guided model/feature selection still requires final held-out evaluation.

## Provenance and limitations

- Dataset SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.
- Split SHA-256: `bccaf24876a5914e2afc03ce3388fcdd0cc8fa6052f9ea52f6c5d38964e29a20`.
- Full metrics, fold assignments, searches, coefficients, predictions and manifest: `reports/runs/logistic_regression_baseline_v2`.
- Feature coefficients/importances, where available, are associations rather than causal effects.
- Raw sources remain absent locally; source-label reproducibility and exact numerical reproduction are not established.
- Balanced sampling probabilities do not estimate commercial success probabilities for natural music releases.

Reproduce with `python -m src.models.train --model logistic_regression`; each run creates separate artifact directories.

Method reference: [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
