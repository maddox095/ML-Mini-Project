# Course-scoped accuracy development

Run: `course_accuracy_v2`.
Only logistic/polynomial logistic regression, decision trees, bagging, random forests and tree boosting were used.
Scores are mean +/- sample SD of five outer artist-disjoint training-validation folds.
Inner CV selects parameters using accuracy, then fitting-data OOF predictions choose a threshold.
Outer validation labels never choose parameters or thresholds. Existing holdout sets were not loaded.

| Model | Default CV accuracy | Threshold-tuned CV accuracy | CV ROC-AUC | Train accuracy |
| --- | ---: | ---: | ---: | ---: |
| random_forest | 82.61% | 82.77% +/- 0.92% | 0.8864 | 83.10% |
| gradient_boosting | 82.54% | 82.54% +/- 1.21% | 0.8850 | 82.84% |
| adaboost | 82.41% | 82.44% +/- 1.13% | 0.8743 | 82.41% |
| bagged_trees | 82.44% | 82.41% +/- 1.14% | 0.8850 | 82.37% |
| decision_tree | 82.34% | 82.34% +/- 1.10% | 0.8483 | 82.34% |
| polynomial_logistic | 82.41% | 82.27% +/- 1.15% | 0.8779 | 82.44% |
| logistic_regression | 82.18% | 82.21% +/- 0.86% | 0.8706 | 82.37% |

Development candidate: **random_forest**, mean outer CV accuracy **82.77%**.
Final fitting-data cutoff: `0.5674968768867616`. Parameters: `{"model__max_depth": 10, "model__max_features": "sqrt", "model__max_samples": 1.0, "model__min_samples_leaf": 5}`.
A fresh independent test set is needed for a new final performance claim. The previously published v1 tests remain unchanged.
These are exploratory development results after inspection of v1 scores; they do not certify 85–90% future accuracy.
Source/data SHA-256: `fe9dd0b5c6e5683e2ca1bd9c3dba69cc9bdef55a987052039cdc7a45e68541a7`.
Actual source snapshot, all search results, inner threshold curves, outer predictions and fold models: `reports/runs/course_accuracy_v2/`, `models/course_accuracy_v2/`.

## Learning curves and data expansion

Fixed-setting curves change from 50% to 100% of fitting artist groups as follows: logistic regression 82.11% to 82.21%, decision tree 82.14% to 82.34%, forest 82.34% to 82.41%. These are development diagnostics with fixed parameters, separate from the nested search above. They suggest limited gains from adding examples with the same four features.

![Training-only learning curves](figures/course_accuracy_v2/learning_curves.png)

The dataset remains 4,000 rows. Registered raw-source restoration failed for Billboard and remained incomplete for MSD, so the 538 additional matched positives cannot be reconstructed from the retained audits. Unverified downloaded bytes were removed. Details: `reports/source_restore_v2_all.json`.

[Complete per-model source and fitted models](course_model_artifacts_v2.md). The original v1 decision tree and published test scores remain unchanged.
