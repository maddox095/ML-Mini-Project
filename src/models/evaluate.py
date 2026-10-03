"""Consistent binary classification and probability metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, make_scorer, precision_score, recall_score, roc_auc_score,
)

SCORING = {
    "roc_auc": "roc_auc", "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "average_precision": "average_precision", "neg_log_loss": "neg_log_loss",
}
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc",
           "average_precision", "log_loss", "brier_score")


def classification_metrics(y, scores, predictions) -> dict:
    """Ranking/classification metrics accept finite raw decision scores."""
    truth = np.asarray(y)
    values = np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or values.shape != truth.shape
            or set(np.unique(truth)) != {0, 1}):
        raise ValueError("Metrics require aligned scores and both binary classes")
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    predicted = np.asarray(predictions)
    if predicted.shape != truth.shape or not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("Predictions must be aligned binary labels")
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, values)),
        "average_precision": float(average_precision_score(truth, values)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def binary_metrics(y, probabilities, *, predictions=None, threshold: float = 0.5) -> dict:
    probability = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    predicted = (probability >= threshold).astype(int) if predictions is None else predictions
    metrics = classification_metrics(y, probability, predicted)
    return {**metrics, "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, probability))}


def prediction_scores(estimator, X) -> tuple[np.ndarray, str]:
    """Keep SVM margins distinct from probabilities; positive class is hit=1."""
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("Estimator must use the binary class order [0, 1]")
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1], "probability"
    return np.asarray(estimator.decision_function(X)), "decision_score"


def score_metrics(y, values, *, predictions, score_kind: str) -> dict:
    if score_kind == "probability":
        return binary_metrics(y, values, predictions=predictions)
    if score_kind == "decision_score":
        return classification_metrics(y, values, predictions)
    raise ValueError(f"Unknown score kind: {score_kind}")


def scoring_for(estimator) -> dict:
    """Do not request probability metrics for estimators without probabilities."""
    return {key: value for key, value in SCORING.items()
            if key != "neg_log_loss" or hasattr(estimator, "predict_proba")}
