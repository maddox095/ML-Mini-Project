# Model setup

The active workflow uses regression, decision trees and ensembles only.
`configs/accuracy_v2.yaml` declares seven completed families and their bounded
searches. [Results](../../reports/course_accuracy_v2.md) and
[complete code/model bundles](../../reports/course_model_artifacts_v2.md).

## Commands

Use a new run ID for every experiment:

```powershell
python -m src.models.accuracy_v2 --run-id course_accuracy_v2_rerun
python -m src.models.export_course --run-id course_accuracy_v2_rerun
python -m src.models.course_replay --evaluate-run course_accuracy_v2 --model random_forest
```

To replay an exact saved setting, run from its extracted bundle:

```powershell
python hyperparameters/setting_0000_train.py
python hyperparameters/setting_0000_test.py
python evaluate_saved_model.py
```

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `data.py` | Hash-verified partitions and explicit final-test access guard |
| `validation.py` | Artist-separated folds, parameter search and replay indices |
| `artifacts.py` | Strict JSON, model save/reload checks and complete source snapshots |
| `bundles.py` | Source copying, import-safe setting programs and content hashes |
| `evaluate.py` | Shared classification and probability metrics |
| `course_models.py` | Allowed estimator factories and deterministic search caps |
| `accuracy_v2.py` | Training-only nested tuning and threshold selection |
| `course_replay.py` | Fixed-setting refits and saved outer-fold evaluation |
| `course_learning_curve.py` | Fixed-parameter development learning curves |
| `export_course.py` | Actual code/data/parameters/models packaged by family |

Preprocessing is fitted inside each training fold. Original test partitions
are never loaded by course training, replay or learning-curve commands.
V2 accuracy is development CV, not independent final-test evidence.

## Earlier v1 compatibility

`train.py`, `registry.py`, `compare.py`, `replay.py` and `finalize.py`
remain for the published v1 benchmark and its saved artifacts. Existing imports
from `train` for shared helpers remain compatible. The default inference CLI
still uses the published v1 tree.

The refactor changes maintained source only. Original fit snapshots, model
weights, test results and prior ZIP bundles are preserved. New runs capture all
refactored source dependencies. [Reuse guide](../../docs/MODEL_REPRODUCIBILITY.md).
