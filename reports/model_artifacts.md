# Complete model artifacts

All seven focused model families are trained and tested. Each ZIP includes actual Python source, the dataset/splits, all tried settings, final pipelines, training results and final test code/results.

| Model | Complete ZIP | Setting/stage records |
| --- | --- | ---: |
| decision_tree | [decision_tree.zip](../models/reproducibility/decision_tree.zip) | 432 |
| logistic_regression | [logistic_regression.zip](../models/reproducibility/logistic_regression.zip) | 180 |
| neural_network | [neural_network.zip](../models/reproducibility/neural_network.zip) | 144 |
| random_forest | [random_forest.zip](../models/reproducibility/random_forest.zip) | 216 |
| svm_linear | [svm_linear.zip](../models/reproducibility/svm_linear.zip) | 144 |
| svm_polynomial | [svm_polynomial.zip](../models/reproducibility/svm_polynomial.zip) | 432 |
| svm_rbf | [svm_rbf.zip](../models/reproducibility/svm_rbf.zip) | 576 |

Total: 2,124 setting/stage records, each with JSON parameters and runnable train/validation scripts.

[Reuse instructions](../docs/MODEL_REPRODUCIBILITY.md). [Final results](final_model_results.md).

The selected deployment pipeline is the original random forest at `models/best_pipeline.joblib`; current configuration, metrics and checksum are in `models/metadata.json`. Original experiment parameters and the historical pre-test selection remain in `reports/frozen_hyperparameters.json`. The forest was promoted after reviewing the completed comparison and records 82.48% primary test accuracy. The original ZIP bundles are preserved unchanged. ZIP bundles and fitted checkpoints are published with Git LFS; run `git lfs pull` after cloning. Expanded bundles and run directories remain local, with experiment records preserved inside the ZIPs.
