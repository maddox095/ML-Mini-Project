# Model comparison in training validation

These runs use matching dataset/split hashes, CV settings, training identities and outer validation folds.
Metrics are nested cross-validation means +/- sample standard deviations. These training runs did not evaluate final test sets; see [the separate final evaluation](final_model_results.md).

| Protocol | Model | Inputs | CV accuracy | CV ROC-AUC | CV F1 |
| --- | --- | --- | ---: | ---: | ---: |
| artist_disjoint_partition | decision_tree | audio_artist_score | 0.819 +/- 0.010 | 0.880 +/- 0.007 | 0.809 |
| artist_disjoint_partition | logistic_regression | audio_artist_score | 0.820 +/- 0.011 | 0.870 +/- 0.015 | 0.798 |
| artist_disjoint_partition | neural_network | audio_artist_score | 0.825 +/- 0.008 | 0.885 +/- 0.008 | 0.804 |
| artist_disjoint_partition | random_forest | audio_artist_score | 0.824 +/- 0.012 | 0.884 +/- 0.011 | 0.802 |
| artist_disjoint_partition | svm_linear | audio_artist_score | 0.823 +/- 0.011 | 0.871 +/- 0.015 | 0.800 |
| artist_disjoint_partition | svm_polynomial | audio_artist_score | 0.823 +/- 0.011 | 0.876 +/- 0.015 | 0.800 |
| artist_disjoint_partition | svm_rbf | audio_artist_score | 0.824 +/- 0.012 | 0.874 +/- 0.015 | 0.801 |
| artist_disjoint_partition | decision_tree | audio_only | 0.657 +/- 0.013 | 0.690 +/- 0.018 | 0.694 |
| artist_disjoint_partition | logistic_regression | audio_only | 0.619 +/- 0.025 | 0.666 +/- 0.023 | 0.647 |
| artist_disjoint_partition | neural_network | audio_only | 0.675 +/- 0.022 | 0.723 +/- 0.019 | 0.714 |
| artist_disjoint_partition | random_forest | audio_only | 0.671 +/- 0.019 | 0.723 +/- 0.014 | 0.717 |
| artist_disjoint_partition | svm_linear | audio_only | 0.623 +/- 0.028 | 0.666 +/- 0.023 | 0.655 |
| artist_disjoint_partition | svm_polynomial | audio_only | 0.664 +/- 0.025 | 0.711 +/- 0.019 | 0.723 |
| artist_disjoint_partition | svm_rbf | audio_only | 0.671 +/- 0.014 | 0.720 +/- 0.017 | 0.728 |
| stratified_partition | decision_tree | audio_artist_score | 0.810 +/- 0.016 | 0.872 +/- 0.008 | 0.796 |
| stratified_partition | logistic_regression | audio_artist_score | 0.822 +/- 0.011 | 0.873 +/- 0.015 | 0.802 |
| stratified_partition | neural_network | audio_artist_score | 0.826 +/- 0.016 | 0.883 +/- 0.011 | 0.804 |
| stratified_partition | random_forest | audio_artist_score | 0.827 +/- 0.017 | 0.886 +/- 0.013 | 0.806 |
| stratified_partition | svm_linear | audio_artist_score | 0.824 +/- 0.017 | 0.872 +/- 0.015 | 0.802 |
| stratified_partition | svm_polynomial | audio_artist_score | 0.824 +/- 0.017 | 0.878 +/- 0.014 | 0.802 |
| stratified_partition | svm_rbf | audio_artist_score | 0.826 +/- 0.015 | 0.871 +/- 0.014 | 0.804 |
| stratified_partition | decision_tree | audio_only | 0.652 +/- 0.013 | 0.690 +/- 0.013 | 0.689 |
| stratified_partition | logistic_regression | audio_only | 0.619 +/- 0.018 | 0.664 +/- 0.020 | 0.644 |
| stratified_partition | neural_network | audio_only | 0.666 +/- 0.023 | 0.713 +/- 0.018 | 0.708 |
| stratified_partition | random_forest | audio_only | 0.671 +/- 0.018 | 0.720 +/- 0.012 | 0.722 |
| stratified_partition | svm_linear | audio_only | 0.621 +/- 0.018 | 0.665 +/- 0.020 | 0.650 |
| stratified_partition | svm_polynomial | audio_only | 0.665 +/- 0.016 | 0.707 +/- 0.012 | 0.725 |
| stratified_partition | svm_rbf | audio_only | 0.670 +/- 0.018 | 0.720 +/- 0.012 | 0.721 |

## Artist-disjoint comparison with all four inputs

- decision_tree versus LR: accuracy difference -0.17 percentage points; ROC-AUC difference +0.0091.
- neural_network versus LR: accuracy difference +0.46 percentage points; ROC-AUC difference +0.0143.
- random_forest versus LR: accuracy difference +0.36 percentage points; ROC-AUC difference +0.0133.
- svm_linear versus LR: accuracy difference +0.30 percentage points; ROC-AUC difference +0.0001.
- svm_polynomial versus LR: accuracy difference +0.30 percentage points; ROC-AUC difference +0.0054.
- svm_rbf versus LR: accuracy difference +0.33 percentage points; ROC-AUC difference +0.0034.
These differences support later model selection; they are not evidence of a final-test winner.

## Provenance

- Dataset SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.
- Split SHA-256: `bccaf24876a5914e2afc03ce3388fcdd0cc8fa6052f9ea52f6c5d38964e29a20`.
- Included runs: `decision_tree_baseline_v1`, `logistic_regression_baseline_v2`, `neural_network_baseline_v1`, `random_forest_baseline_v1`, `linear_svm_baseline_v1`, `svm_polynomial_baseline_v1`, `svm_rbf_baseline_v1`.
- Artist-Score-only results are diagnostic and remain in each model's individual report.
- Probability-quality metrics are available for LR, trees, forests and NN; uncalibrated SVM margins are not probabilities.
- Raw sources remain absent locally; this is preliminary Mode B methodology evidence.
