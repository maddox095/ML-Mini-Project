# Selected random forest and recorded model comparison

Current inference model: **random_forest / audio_artist_score**, version **random_forest_v1_deployment**.

The original random forest has the highest observed accuracy among the original classifiers on the 976-song artist-disjoint test partition: **82.48% accuracy, 92.64% precision, 70.25% recall, F1 0.7991 and ROC-AUC 0.8767**. The selected model has 300 trees, maximum depth 8, minimum leaf size 5 and square-root feature sampling. It uses the same saved four-input preprocessing pipeline in the CLI and demo source.

The forest was chosen for deployment after reviewing the completed comparison. This is not a new independent test or a claim that it was selected before testing. Original run records retain the historical decision-tree selection. The neural network has the highest observed test ROC-AUC; the current deployment choice prioritizes accuracy.

| Protocol | Model | Features | Deployed now | Historical pre-test selection | Test accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| artist_disjoint_partition | decision_tree | audio_artist_score |  | yes | 0.7920 | 0.8172 | 0.7479 | 0.7810 | 0.8747 | 0.8667 |
| artist_disjoint_partition | decision_tree | audio_only |  |  | 0.6291 | 0.5968 | 0.7769 | 0.6750 | 0.6789 | 0.6149 |
| artist_disjoint_partition | dummy | prior |  |  | 0.4959 | 0.4959 | 1.0000 | 0.6630 | 0.5000 | 0.4959 |
| artist_disjoint_partition | logistic_regression | audio_artist_score |  |  | 0.8145 | 0.9084 | 0.6963 | 0.7883 | 0.8668 | 0.8698 |
| artist_disjoint_partition | logistic_regression | audio_only |  |  | 0.6086 | 0.5867 | 0.7128 | 0.6437 | 0.6549 | 0.6018 |
| artist_disjoint_partition | neural_network | audio_artist_score |  |  | 0.8207 | 0.9164 | 0.7025 | 0.7953 | 0.8863 | 0.8845 |
| artist_disjoint_partition | neural_network | audio_only |  |  | 0.6701 | 0.6274 | 0.8244 | 0.7125 | 0.7228 | 0.6673 |
| artist_disjoint_partition | random_forest | audio_artist_score | yes |  | 0.8248 | 0.9264 | 0.7025 | 0.7991 | 0.8767 | 0.8637 |
| artist_disjoint_partition | random_forest | audio_only |  |  | 0.6455 | 0.6068 | 0.8099 | 0.6938 | 0.7079 | 0.6461 |
| artist_disjoint_partition | svm_linear | audio_artist_score |  |  | 0.8156 | 0.9108 | 0.6963 | 0.7892 | 0.8668 | 0.8696 |
| artist_disjoint_partition | svm_linear | audio_only |  |  | 0.6137 | 0.5887 | 0.7335 | 0.6532 | 0.6551 | 0.6023 |
| artist_disjoint_partition | svm_polynomial | audio_artist_score |  |  | 0.8156 | 0.9108 | 0.6963 | 0.7892 | 0.8776 | 0.8799 |
| artist_disjoint_partition | svm_polynomial | audio_only |  |  | 0.6670 | 0.6154 | 0.8760 | 0.7229 | 0.7100 | 0.6617 |
| artist_disjoint_partition | svm_rbf | audio_artist_score |  |  | 0.8176 | 0.9158 | 0.6963 | 0.7911 | 0.8724 | 0.8764 |
| artist_disjoint_partition | svm_rbf | audio_only |  |  | 0.6557 | 0.6175 | 0.8037 | 0.6984 | 0.7142 | 0.6591 |
| stratified_partition | decision_tree | audio_artist_score |  |  | 0.8070 | 0.8866 | 0.7040 | 0.7848 | 0.8614 | 0.8624 |
| stratified_partition | decision_tree | audio_only |  |  | 0.6510 | 0.6208 | 0.7760 | 0.6898 | 0.7008 | 0.6335 |
| stratified_partition | dummy | prior |  |  | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| stratified_partition | logistic_regression | audio_artist_score |  |  | 0.8060 | 0.9026 | 0.6860 | 0.7795 | 0.8643 | 0.8697 |
| stratified_partition | logistic_regression | audio_only |  |  | 0.6280 | 0.6127 | 0.6960 | 0.6517 | 0.6699 | 0.6120 |
| stratified_partition | neural_network | audio_artist_score |  |  | 0.8140 | 0.9266 | 0.6820 | 0.7857 | 0.8812 | 0.8803 |
| stratified_partition | neural_network | audio_only |  |  | 0.6680 | 0.6288 | 0.8200 | 0.7118 | 0.7214 | 0.6552 |
| stratified_partition | random_forest | audio_artist_score |  |  | 0.8160 | 0.9389 | 0.6760 | 0.7860 | 0.8795 | 0.8734 |
| stratified_partition | random_forest | audio_only |  |  | 0.6640 | 0.6206 | 0.8440 | 0.7153 | 0.7230 | 0.6592 |
| stratified_partition | svm_linear | audio_artist_score |  |  | 0.8130 | 0.9311 | 0.6760 | 0.7833 | 0.8641 | 0.8696 |
| stratified_partition | svm_linear | audio_only |  |  | 0.6310 | 0.6127 | 0.7120 | 0.6586 | 0.6693 | 0.6119 |
| stratified_partition | svm_polynomial | audio_artist_score |  |  | 0.8130 | 0.9311 | 0.6760 | 0.7833 | 0.8752 | 0.8779 |
| stratified_partition | svm_polynomial | audio_only |  |  | 0.6620 | 0.6128 | 0.8800 | 0.7225 | 0.7248 | 0.6713 |
| stratified_partition | svm_rbf | audio_artist_score |  |  | 0.8130 | 0.9311 | 0.6760 | 0.7833 | 0.8693 | 0.8737 |
| stratified_partition | svm_rbf | audio_only |  |  | 0.6680 | 0.6183 | 0.8780 | 0.7256 | 0.7308 | 0.6748 |

## Frozen hyperparameters

These values were selected on full training CV before either holdout was opened.
All fixed estimator parameters are also saved in `reports/frozen_hyperparameters.json` and each model bundle.

| Protocol | Model | Features | Tuned parameters |
| --- | --- | --- | --- |
| stratified_partition | decision_tree | audio_only | `{"max_depth": 8, "min_samples_leaf": 20}` |
| stratified_partition | decision_tree | audio_artist_score | `{"max_depth": 8, "min_samples_leaf": 20}` |
| artist_disjoint_partition | decision_tree | audio_only | `{"max_depth": 8, "min_samples_leaf": 20}` |
| artist_disjoint_partition | decision_tree | audio_artist_score | `{"max_depth": 8, "min_samples_leaf": 20}` |
| stratified_partition | svm_linear | audio_only | `{"C": 0.1}` |
| stratified_partition | svm_linear | audio_artist_score | `{"C": 0.1}` |
| artist_disjoint_partition | svm_linear | audio_only | `{"C": 0.1}` |
| artist_disjoint_partition | svm_linear | audio_artist_score | `{"C": 1.0}` |
| stratified_partition | logistic_regression | audio_only | `{"C": 1.0}` |
| stratified_partition | logistic_regression | audio_artist_score | `{"C": 0.01}` |
| artist_disjoint_partition | logistic_regression | audio_only | `{"C": 1.0}` |
| artist_disjoint_partition | logistic_regression | audio_artist_score | `{"C": 100.0}` |
| stratified_partition | neural_network | audio_only | `{"alpha": 0.01}` |
| stratified_partition | neural_network | audio_artist_score | `{"alpha": 0.01}` |
| artist_disjoint_partition | neural_network | audio_only | `{"alpha": 0.01}` |
| artist_disjoint_partition | neural_network | audio_artist_score | `{"alpha": 0.0001}` |
| stratified_partition | random_forest | audio_only | `{"max_depth": 4, "max_features": "sqrt", "min_samples_leaf": 5}` |
| stratified_partition | random_forest | audio_artist_score | `{"max_depth": 4, "max_features": "sqrt", "min_samples_leaf": 5}` |
| artist_disjoint_partition | random_forest | audio_only | `{"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 5}` |
| artist_disjoint_partition | random_forest | audio_artist_score | `{"max_depth": 8, "max_features": "sqrt", "min_samples_leaf": 5}` |
| stratified_partition | svm_polynomial | audio_only | `{"C": 1.0, "coef0": 1.0, "degree": 2, "gamma": "scale"}` |
| stratified_partition | svm_polynomial | audio_artist_score | `{"C": 0.1, "coef0": 1.0, "degree": 3, "gamma": 0.01}` |
| artist_disjoint_partition | svm_polynomial | audio_only | `{"C": 1.0, "coef0": 1.0, "degree": 2, "gamma": "scale"}` |
| artist_disjoint_partition | svm_polynomial | audio_artist_score | `{"C": 0.1, "coef0": 1.0, "degree": 3, "gamma": 0.01}` |
| stratified_partition | svm_rbf | audio_only | `{"C": 10.0, "gamma": 0.1}` |
| stratified_partition | svm_rbf | audio_artist_score | `{"C": 0.1, "gamma": 0.01}` |
| artist_disjoint_partition | svm_rbf | audio_only | `{"C": 0.1, "gamma": 1.0}` |
| artist_disjoint_partition | svm_rbf | audio_artist_score | `{"C": 0.1, "gamma": 0.01}` |

## Selected model evidence

Selected forest confusion matrix: TN=465, FP=27, FN=144, TP=340.
Saved model: `models/best_pipeline.joblib`; metadata: `models/metadata.json`.
Historical selection, unchanged comparison metrics, predictions and hashes: `reports/runs/final_suite_v1/`.
Combined train/CV/test metrics: `reports/model_comparison.csv`.
Selected forest confusion matrix: `reports/figures/random_forest_v1_deployment/selected_confusion_matrix.png`. Original comparison curves: `reports/figures/final_suite_v1/`.

## Limits

Raw source files are absent, so source labels and the full extraction cannot be independently reproduced locally.
This is preliminary four-input Mode B evidence, not numerical reproduction of the original study.
The balanced research sample does not yield natural-release commercial success probabilities.
SVM scores are uncalibrated margins; their log loss and Brier scores are intentionally absent.
Original pipelines were fitted only on their training partitions. The current forest deployment was chosen after reviewing the completed comparison, with no refitting and no fresh independent evaluation. Historical pre-test selection remains unchanged in the archived records.
