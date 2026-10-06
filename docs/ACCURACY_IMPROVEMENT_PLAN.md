# Accuracy improvement within the course syllabus

The target is 85-90% accuracy on unseen artist keys. It is an experimental
target, not a promised result. New experiments use regression, decision trees
and ensembles only.

## Implemented accuracy experiment

The original decision tree was selected using ROC-AUC and a simplicity rule,
then achieved 79.20% on the original artist-disjoint test set. The original
forest achieved 82.48% on that test set. Those results remain the v1 benchmark. The saved original forest is now the selected deployment model for the CLI and demo source, version `random_forest_v1_deployment`. Its promotion followed review of the completed comparison; it is not a fresh independent test. The original tree selection is preserved as historical evidence.

`configs/accuracy_v2.yaml` declares seven allowed model families:

1. Regularized logistic regression, with six regularization strengths.
2. Degree-two polynomial features followed by logistic regression, with the
   transformations and scaling fitted inside each training fold.
3. Decision trees, searching depth, minimum leaf size and pruning strength.
4. Bagged decision trees, searching base-tree settings and sample fraction.
5. Random forests, searching depth, leaf size, feature and sample fraction.
6. AdaBoost using small decision trees, searching tree depth, learning rate
   and number of estimators.
7. Gradient boosting with trees, searching depth, leaf size, learning rate,
   estimator count and sample fraction.

Because the target is binary classification, the regression models are
logistic regression rather than ordinary least-squares regression.
Deterministic, capped candidate grids make the search practical on CPU.

`src/models/accuracy_v2.py` uses the original 3,024-row training partition only.
All families share the same five outer artist-disjoint validation folds.
Five inner folds choose parameters using accuracy. Inner out-of-fold
probabilities then choose an accuracy threshold; each outer validation fold
evaluates that threshold without selecting it. The threshold tie rule is
closest to 0.5, then the lower threshold. Exact family-score ties follow
the declared family order.

The experiment saves all candidate scores, threshold curves, outer predictions,
full parameters, source snapshots and six fitted pipelines per family (five
outer fits and one full training fit). Per-model bundles include actual source
and runnable training/validation scripts for every tried setting. Discarded
inner-trial weights can be regenerated with the supplied code.

See [v2 results](../reports/course_accuracy_v2.md) and
[saved v2 code](../reports/course_model_artifacts_v2.md).

## Data checks and learning curves

The 4,000-row table remains unchanged. Its four inputs are tempo, loudness,
duration and a verified binary prior-hit indicator. Identity and chart outcome
metadata are excluded. The existing handoff checks verify completeness,
finite values, unique identities, chart chronology and artist-separated splits.

Original models cluster near 82-83% training/validation accuracy. Artist Score
alone gives about 82.34% validation accuracy; the strongest original audio-only
mean was approximately 67.5%. This suggests that the current audio descriptors
provide limited additional information. It does not prove an accuracy ceiling.

`src/models/course_learning_curve.py` tests fixed logistic regression, tree
and forest settings on 25%, 50%, 75% and 100% of fitting artist groups. The same
outer validation artists are held aside at every size; no original test rows
are loaded. These development curves describe the value of more examples
under the current features, rather than guarantee gains from a larger dataset.
Results are saved in `reports/runs/course_learning_curve_v2/`.

The matching audit records 2,538 positive audio-feature matches before sampling
and 8,340 unresolved Billboard identities. Current audits cannot reconstruct
the audio features of the 538 unsampled positives. The registered raw sources
must be restored and hash-verified before any versioned expansion. The source
restoration attempt and its limitations are in
`reports/source_restore_v2_all.json`. No missing features, counts or labels
are manufactured. Unmatched negatives remain non-hit candidates, rather than
verified never-hits.

Improving the dataset within these methods means verifying more song matches,
auditing ambiguous labels and adding properly documented historical features
only when the underlying source and reference dates are available. The current
negative reference dates are synthetic January 1 dates, so recency cannot be
silently treated as an equivalent measured feature for both classes.

## A new final performance claim

The original test results have already been inspected. V2 scores are development
cross-validation results, not a fresh final test score. Selecting the best mean
among seven families also makes its headline score a development estimate.
A new 85-90% final claim requires independently acquired songs/artists or another
genuinely untouched set reserved before further development. Reshuffling the
same previously evaluated rows does not create an untouched test set.

The original [v1 results](../reports/final_model_results.md), fitted model and
[code bundles](../reports/model_artifacts.md) remain available.
