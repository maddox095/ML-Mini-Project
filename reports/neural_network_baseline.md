# Six-unit Neural Network baseline

Run: `neural_network_baseline_v1`. Device: CPU. Status: preliminary Mode B experiment.

Scores below are mean +/- sample standard deviation across 5 outer training-validation folds.
Each Six-unit Neural Network fold tunes its parameters in 5 separate inner folds. The final test sets were not evaluated.

| Protocol | Model | Features | CV accuracy | CV ROC-AUC | CV F1 | Selected C |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| stratified_partition | dummy | prior | 0.500 +/- 0.000 | 0.500 +/- 0.000 | 0.000 | - |
| stratified_partition | neural_network | audio_only | 0.666 +/- 0.023 | 0.713 +/- 0.018 | 0.708 | - |
| stratified_partition | neural_network | audio_artist_score | 0.826 +/- 0.016 | 0.883 +/- 0.011 | 0.804 | - |
| stratified_partition | neural_network | artist_score_only | 0.824 +/- 0.017 | 0.824 +/- 0.017 | 0.802 | - |
| artist_disjoint_partition | dummy | prior | 0.501 +/- 0.000 | 0.500 +/- 0.000 | 0.668 | - |
| artist_disjoint_partition | neural_network | audio_only | 0.675 +/- 0.022 | 0.723 +/- 0.019 | 0.714 | - |
| artist_disjoint_partition | neural_network | audio_artist_score | 0.825 +/- 0.008 | 0.885 +/- 0.008 | 0.804 | - |
| artist_disjoint_partition | neural_network | artist_score_only | 0.823 +/- 0.011 | 0.824 +/- 0.011 | 0.800 | - |

## Candidate for later comparison

The current Six-unit Neural Network candidate uses `audio_artist_score` under the artist-disjoint protocol.
Pipeline: `models/neural_network_baseline_v1/artist_disjoint_partition__neural_network__audio_artist_score.joblib`. This is a baseline candidate, not the final model for deployment.
Final refit parameters are selected by CV on all outer training rows; outer folds tune independently.

## Training-only diagnostics

- fixed_parameters_seed_stability: mean accuracy 0.820; mean ROC-AUC 0.880, SD across seed means 0.0054.
- shuffled_training_labels: mean accuracy 0.511; mean ROC-AUC 0.504, SD across seed means 0.0211.

All 8 saved pipelines passed save/reload score and class agreement checks.
Audio-only and Artist-Score-only comparisons quantify the influence of artist history.
These are training-validation results; CV-guided model/feature selection still requires final held-out evaluation.

## Provenance and limitations

- Dataset SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.
- Split SHA-256: `bccaf24876a5914e2afc03ce3388fcdd0cc8fa6052f9ea52f6c5d38964e29a20`.
- Full metrics, fold assignments, searches, coefficients, predictions and manifest: `reports/runs/neural_network_baseline_v1`.
- Feature coefficients/importances, where available, are associations rather than causal effects.
- Raw sources remain absent locally; source-label reproducibility and exact numerical reproduction are not established.
- Balanced sampling probabilities do not estimate commercial success probabilities for natural music releases.

Reproduce with `python -m src.models.train --model neural_network`; each run creates separate artifact directories.

The network has one hidden layer of six sigmoid units and a binary sigmoid output with L2 regularization.
L-BFGS replaces the proposed Adam/epoch-checkpoint schedule. There is no internal random validation split.
The optimizer and iteration budget are recorded methodology adaptations; iterations are not Adam epochs.
