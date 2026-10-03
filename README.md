# HitPredict

Hit-song classification using Million Song Dataset audio descriptors and
historical Billboard labels. The current dataset has **4,000 songs: 2,000 hits
and 2,000 non-hit candidates**, with four inputs: tempo, loudness, duration and
a binary prior-hit Artist Score.

The active User 2 workflow uses only **regression, decision trees and
ensembles**. Seven families are trained: logistic regression, degree-two
polynomial logistic regression, decision trees, bagged trees, random forests,
AdaBoost and gradient boosting.

The best development result is **random forest: 82.77% +/- 0.92%** across five
artist-disjoint validation folds. None reached 85%. These scores are training
cross-validation results, not a fresh final test score.
[Results](reports/course_accuracy_v2.md) and
[complete model/code bundles](reports/course_model_artifacts_v2.md).

## Setup and checks

Use Python 3.12 and the preserved dependency lock:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.data.verify_handoff
.\.venv\Scripts\python.exe -B scripts/verify_artifacts.py
```

Artifact verification is read-only: it checks datasets, original source
snapshots, saved models, results and ZIPs without fitting or loading test rows.
Tests are restricted to `tests/`; saved replay programs are not collected.

## Train and reuse models

New runs require a unique run ID and never overwrite previous experiments:

```powershell
.\.venv\Scripts\python.exe -m src.models.accuracy_v2 --run-id course_accuracy_v2_rerun
.\.venv\Scripts\python.exe -m src.models.export_course --run-id course_accuracy_v2_rerun
```

`configs/accuracy_v2.yaml` controls all seven families, bounded searches, seeds
and validation. The runner loads only the original 3,024 training rows, keeps
artist groups separate at both CV levels, fits preprocessing within training,
and selects thresholds without using outer validation labels. Training uses CPU.

The completed bundles contain **420 setting/stage records, actual training and
evaluation code, and 42 fitted pipelines**. Each bundle saves full fixed and
tuned parameters, five outer-fold checkpoints and the final training model.
From an extracted bundle with the locked dependencies installed:

```powershell
python hyperparameters/setting_0000_train.py
python hyperparameters/setting_0000_test.py
python evaluate_saved_model.py
```

The testing commands reproduce inner or outer validation results.
[Reuse instructions](docs/MODEL_REPRODUCIBILITY.md).

Fitted checkpoints and all 14 complete model ZIPs are published with Git LFS.
Install Git LFS, then run `git lfs install` and `git lfs pull` after cloning.
Extract a ZIP to use its preserved experiment code and data. Expanded bundle
folders and run directories remain local; their records are included in ZIPs.

## Dataset and source limitations

`data/interim/model_table.parquet` has 2,660 distinct normalized artist keys,
complete finite inputs and unique song identities. Identifiers, names, years,
reference dates and chart outcomes remain metadata. Artist Score uses earlier
chart history and excludes the current song.

The stratified split has 3,000 training/1,000 test rows; the primary
artist-disjoint split has 3,024 training/976 test rows with no shared normalized
artist keys. V2 uses the primary training partition only.

The registered raw MSD and Billboard sources are unavailable locally. Their
restoration attempt did not produce verified files, so expansion and a full
source rebuild remain blocked. The retained handoff passes 27 integrity checks;
that does not independently verify all source labels. Non-hit candidates are
not proof of never charting. The balanced research sample does not establish
natural-release commercial-success probabilities.

[Data handoff](docs/USER2_HANDOFF.md),
[verification](reports/user1_verification.md), and
[accuracy improvement plan](docs/ACCURACY_IMPROVEMENT_PLAN.md).

## Earlier published benchmark and inference

The original v1 benchmark and its complete code bundles remain preserved.
Its CV-selected decision tree achieved **79.20% held-out accuracy** on unseen
artist keys. [Published v1 results](reports/final_model_results.md) and
[v1 bundles](reports/model_artifacts.md).

The existing CLI still uses that frozen decision tree:

```powershell
.\.venv\Scripts\python.exe -m src.inference --tempo 120 --loudness -8 --duration 210 --artist-score 1
```

The new forest is saved separately as a development candidate. The browser
demo and final submission report remain pending.
[User 2 workflow](docs/USER2_PLAN.md).

## Streamlit demo and deployment

The assignment-aligned browser app is `app/app.py`. It loads the published v1
decision tree, verifies its checksum, and shows the model version, predicted
class, balanced-sample probability and artist-disjoint held-out evidence.

```bash
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

For public deployment, use Streamlit Community Cloud with repository
`maddox095/ML-Mini-Project`, branch `main`, and entry point `app/app.py`.
The model artifacts use Git LFS, which Community Cloud supports. See
[`app/README.md`](app/README.md) for the complete deployment steps.

## Code layout

| Path | Responsibility |
| --- | --- |
| `src/data/` | Acquisition, extraction, dataset checks and splits |
| `src/features/` | Identity normalization and input preprocessing |
| `src/models/validation.py` | Shared folds, parameter search and replay row selection |
| `src/models/artifacts.py` | Model serialization, JSON and full source snapshots |
| `src/models/bundles.py` | Shared code copying, replay programs and bundle hashes |
| `src/models/accuracy_v2.py` | Active course-approved nested training workflow |
| `src/models/course_models.py` | Allowed model factories and bounded grids |
| `src/models/course_replay.py` | Replay settings and evaluate saved validation models |
| `src/models/course_learning_curve.py` | Fixed-setting artist-group learning curves |
| `src/models/export_course.py` | Complete per-family reproducibility bundles |
| `src/inference.py` | Frozen v1 prediction CLI |
| `scripts/verify_artifacts.py` | Read-only saved-artifact verification |
| `tests/` | Data, leakage boundaries, replay and serialization checks |

Earlier v1 training, comparison and finalization commands remain compatible.
Original PDFs and author data are references under `docs/` and
`data/reference/author_archive/`; they are not active training inputs.
