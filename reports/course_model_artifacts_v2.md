# Course-scoped model code and artifacts

Seven families completed. Each ZIP contains actual source, data/splits, complete parameters, every tried setting with runnable training/validation scripts, five outer-fold models and the final fitted training pipeline.

| Family | ZIP | Setting/stage records |
| --- | --- | ---: |
| logistic_regression | [logistic_regression.zip](../models/reproducibility_v2/course_accuracy_v2/logistic_regression.zip) | 36 |
| polynomial_logistic | [polynomial_logistic.zip](../models/reproducibility_v2/course_accuracy_v2/polynomial_logistic.zip) | 36 |
| decision_tree | [decision_tree.zip](../models/reproducibility_v2/course_accuracy_v2/decision_tree.zip) | 108 |
| bagged_trees | [bagged_trees.zip](../models/reproducibility_v2/course_accuracy_v2/bagged_trees.zip) | 48 |
| random_forest | [random_forest.zip](../models/reproducibility_v2/course_accuracy_v2/random_forest.zip) | 72 |
| adaboost | [adaboost.zip](../models/reproducibility_v2/course_accuracy_v2/adaboost.zip) | 48 |
| gradient_boosting | [gradient_boosting.zip](../models/reproducibility_v2/course_accuracy_v2/gradient_boosting.zip) | 72 |

Total: 420 setting/stage records and 42 saved fitted pipelines.

The original fit files are verified against source snapshots; subsequent replay/testing code is included as actual Python files. Full fixed and selected pipeline configurations include preprocessing and base-tree parameters.

Run `python hyperparameters/setting_0000_train.py` or `python hyperparameters/setting_0000_test.py` from a bundle. The latter evaluates an inner training-validation fold. `python evaluate_saved_model.py` scores all five saved outer-fold models.

These are development validation results, not a new final test. The original test partitions are never loaded by these commands. Original v1 artifacts remain available separately.

[Results](course_accuracy_v2.md). [Course-only improvement plan](../docs/ACCURACY_IMPROVEMENT_PLAN.md).

The seven ZIPs and fitted checkpoints are published with Git LFS. Run `git lfs pull` after cloning, then extract the desired ZIP. Expanded bundle folders remain local.
