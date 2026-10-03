# User 2 workflow

The active scope is regression, decision trees and ensembles. The earlier v1
benchmark is preserved separately; its historical model list does not define
new work.

## Current progress

| Stage | Status | Evidence |
| --- | --- | --- |
| U2-S1: data verification and preprocessing | Complete | 4,000-row handoff, 27 checks, hashed splits and training-only preprocessing |
| U2-S2: model implementations | Complete | Logistic and polynomial logistic regression, tree, bagging, forest, AdaBoost, gradient boosting |
| U2-S3: tuning and model comparison | Complete | Five inner/five outer artist-group folds, accuracy selection and nested threshold fitting |
| U2-S4: diagnostics | Complete for current development | Learning curves; earlier v1 feature ablation, seed and shuffled-label checks |
| U2-S5: evaluation and error analysis | V1 complete; v2 validation complete | V1 frozen test results; v2 outer predictions, metrics and learning curves |
| U2-S6: model/code preservation | Complete | 42 fitted v2 pipelines, 420 setting/stage records and seven code ZIPs |
| U2-S7: application and submission | In progress | Streamlit demo is ready locally; capture examples and deploy or record it |

The strongest v2 development candidate is **random forest: 82.77% +/- 0.92%
accuracy** across five artist-disjoint validation folds. The 85-90% target
has not been reached. V2 has no fresh independent test score; the old test
results have already been observed.

[Current results](../reports/course_accuracy_v2.md),
[complete code and model bundles](../reports/course_model_artifacts_v2.md),
[reproduction guide](MODEL_REPRODUCIBILITY.md).

## Data and training contract

- Dataset: 4,000 rows, balanced labels, 2,660 normalized artist keys.
- Inputs: tempo (BPM), loudness (dB), duration (seconds), prior-hit Artist Score.
- Target: binary `hit`; other columns remain metadata.
- Primary training rows: 3,024; existing artist-disjoint test rows: 976.
- Five inner folds select parameters by accuracy; inner out-of-fold scores
  select each threshold. Outer validation labels do not select either.
- The threshold tie rule is closest to 0.5, then lower. Exact family-score
  ties use the order in `configs/accuracy_v2.yaml`.
- Every new run gets a unique ID and saves source, full parameters, results,
  predictions, folds, selected thresholds and fitted pipelines.
- The retained data and published v1 artifacts remain unchanged.

The models use bounded searches declared in `configs/accuracy_v2.yaml`.
The shared implementation is described in [model setup](../src/models/README.md).

## Remaining delivery work

1. Run `streamlit run app/app.py` to use the local browser demo. It uses the
   published v1 tree, verifies its saved checksum and displays held-out model
   evidence. Deploy it on Streamlit Community Cloud if a public URL is wanted.
2. Show predicted class, research-sample probability, model version and the
   corresponding test or validation evidence. Verify example predictions.
3. Capture two examples that change at least two inputs, with screenshots
   or a short recording.
4. Prepare the final submission report covering model comparison, methods,
   parameters, preprocessing, results, error analysis and reproducibility.
5. Explain source/label limitations and distinguish v1 held-out scores from
   v2 development CV scores. Keep all new modeling within the syllabus.

Public hosting is optional. A working local demo is sufficient for this stage.

## Accuracy and data limitations

Learning curves show little improvement near the current training size with
these four inputs. Adding verified matches or stronger documented historical
features requires raw source restoration. Billboard retrieval failed and the
MSD transfer remained incomplete, so dataset expansion is blocked. No source
features or labels are invented.

A credible new final score requires independently acquired songs/artists or
another genuinely untouched evaluation set, reserved before further
development. Re-shuffling old test rows does not create a fresh test.

[Course-only improvement plan](ACCURACY_IMPROVEMENT_PLAN.md),
[published v1 benchmark](../reports/final_model_results.md),
[data handoff](USER2_HANDOFF.md).
