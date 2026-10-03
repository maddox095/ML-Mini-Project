# Saved model code and experiments

Every model has a separate complete bundle in `models/reproducibility/<model>/`.
Its matching `<model>.zip` archive contains the same files.
These folders contain actual source files, data, hyperparameters, fitted model
checkpoints and results. Hashes are additional integrity evidence.

## What each bundle contains

| Path within a bundle | Contents |
| --- | --- |
| `src/` | Full current Python training, preprocessing, scoring, testing and replay code |
| `original_training_code/` | Full source tree with the exact recorded fit implementation overlaid; original fit files are verified byte for byte |
| `configs/` | Dataset rules and all model/search settings |
| `requirements-lock.txt` | Pinned runtime and plotting dependencies |
| `data/` and audit CSVs | Current 4,000-row table, exact splits, provenance manifest and integrity evidence |
| `models/<run_id>/` | Saved pipelines including preprocessing, plus model metadata |
| `reports/runs/<run_id>/` | Run manifest, all searches, fold metrics, training predictions and robustness checks |
| `reports/runs/final_suite_v1/` | Frozen suite selection, held-out predictions/metrics and actual final testing source |
| `hyperparameters/setting_NNNN.json` | Complete fixed and varied estimator parameters, run configuration, protocol, feature subset, tuning stage and individual inner-fold results |
| `hyperparameters/setting_NNNN_train.py` | Runnable code to train this precise setting on its recorded training stage |
| `hyperparameters/setting_NNNN_test.py` | Runnable code to reproduce this setting's first inner validation fold |
| `hyperparameter_index.json` | Index of every tried setting and its training stage |
| `evaluate_saved_model.py` | This family's frozen holdout-testing code |
| `bundle_manifest.json` | Integrity hashes for the actual bundle files |

A setting is identified by model, protocol, feature set, tuning stage and
parameters. Outer stages fit only their own training rows. Full-training
tuning uses all rows in the original training partition. All inner-fold
scores and ranks remain in `search_results.csv`; replay reconstructs their
exact fold identities using recorded seeds and the retained split table.

Final tuned pipelines are saved for each feature/protocol combination.
Weights for every discarded inner search fit were not retained; the saved
code and setting files regenerate them. Neural network checkpoints contain
weights, iteration count and loss. L-BFGS iterations are not Adam epochs.

## Reuse a model bundle

Copy its entire folder to the desired machine. In Python 3.12, from that folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe hyperparameters/setting_0000_train.py
.\.venv\Scripts\python.exe hyperparameters/setting_0000_test.py
.\.venv\Scripts\python.exe -m src.models.replay --setting hyperparameters/setting_0000.json --inner-fold 2 --output replay_fold2
.\.venv\Scripts\python.exe evaluate_saved_model.py
```

The per-setting test script scores training-validation rows. Final testing is
separate and uses the already frozen selection and saved checkpoints; it does
not fit or select models. Repeating it reproduces the published experiment,
and must not be used to choose new hyperparameters from held-out scores.

To rerun a full family, use `python -m src.models.train --model <model>`.
It creates a new run rather than overwriting old artifacts. The code in
`original_training_code/` preserves original fit versions, while the bundle's
root `src/` provides the maintained replay and testing tools. The original
source snapshot is also retained under the training run's reports folder.

## Repository commands

```powershell
.\.venv\Scripts\python.exe -m src.models.train --model svm_rbf
.\.venv\Scripts\python.exe -m src.models.finalize --run-id final_suite_v1
.\.venv\Scripts\python.exe -m src.models.export
.\.venv\Scripts\python.exe -m src.inference --tempo 120 --loudness -8 --duration 210 --artist-score 1
```

Finalization requires all seven compatible completed families. It freezes
model/feature/parameter/threshold choices before loading either test set and
refuses to overwrite an existing final evaluation. The final inference
checkpoint is `models/best_pipeline.joblib` with `models/metadata.json`.
Readable frozen parameters are also saved in `reports/frozen_hyperparameters.json`.
The final run records a recovery from a tuple/list JSON comparison failure;
the original selection was preserved and no test data was accessed before the
correction. Both the initial and corrected testing source files are retained.

All saved models are scikit-learn CPU estimators. The neural network has six
sigmoid hidden units and uses L-BFGS with L2 regularization. No GPU training
has been performed. Exact numerical identity across different hardware is
not promised; the pinned environment, data and code enable reproducible
experiments. The raw source archives remain unavailable, and retained data
does not establish independent source-label verification.

Fitted checkpoints, metadata and all 14 complete ZIP bundles are published
on GitHub. Checkpoints and ZIPs use Git LFS: install Git LFS and run
`git lfs install` and `git lfs pull` after cloning. Extract a ZIP before
running its reproduction commands. Expanded bundle folders and
`reports/runs/` remain local; their records are preserved inside the ZIPs.

The repository split CSV now uses LF line endings. Historical ZIPs retain
the original CRLF CSV and matching checksums to preserve each experiment.
The assignments are identical. The artifact verifier checks archive hashes,
all recorded member hashes and agreement with the current LF assignments.
## Course-scoped v2 bundles

The additional accuracy suite has seven complete ZIPs in
`models/reproducibility_v2/course_accuracy_v2/`; see the
[artifact index](../reports/course_model_artifacts_v2.md).
It contains 420 actual setting/stage JSON records and corresponding runnable
training/validation scripts, all source/configuration files, full nested
pipeline parameters and 42 fitted models including selected thresholds.

From an extracted bundle with the locked Python 3.12 dependencies installed:

```powershell
python hyperparameters/setting_0000_train.py
python hyperparameters/setting_0000_test.py
python evaluate_saved_model.py
```

The testing commands evaluate inner or outer training-validation folds.
Original test sets are not loaded. Copied code was verified to reproduce one
inner fit per family and all 35 saved outer-fold scores. Actual training source
files are byte-preserved from the original run snapshot. The learning-curve
code, results and source-restoration limitation are also included.

Original v1 bundles and their published test results remain separate above.
