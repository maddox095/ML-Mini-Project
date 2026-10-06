"""Standalone scientific figures for the frozen final comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve

from src.data.common import repo_path


def plot_deployment(metadata):
    """Plot the deployed model's recorded confusion matrix without evaluation."""
    output = repo_path(f"reports/figures/{metadata['final_run_id']}")
    output.mkdir(parents=True, exist_ok=True)
    values = metadata["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(np.array([[values["tn"], values["fp"]],
                                    [values["fn"], values["tp"]]], dtype=int),
        display_labels=["Non-hit candidate", "Hit"]).plot(ax=ax, colorbar=False)
    ax.set_title("Selected random forest\nRecorded artist-disjoint evaluation")
    fig.tight_layout()
    fig.savefig(output / "selected_confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_final(selection, results, predictions):
    output = repo_path(f"reports/figures/{selection['run_id']}")
    output.mkdir(parents=True, exist_ok=True)
    for protocol in selection["protocols"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        subset = predictions.loc[predictions.protocol.eq(protocol) & predictions.feature_set.eq("audio_artist_score")]
        for model, group in subset.groupby("model"):
            fpr, tpr, _ = roc_curve(group.hit, group.score)
            precision, recall, _ = precision_recall_curve(group.hit, group.score)
            axes[0].plot(fpr, tpr, label=model)
            axes[1].plot(recall, precision, label=model)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC: all four inputs")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall: all four inputs")
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle(protocol + " — frozen held-out predictions")
        fig.tight_layout()
        fig.savefig(output / f"{protocol}__roc_pr.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5))
        for model, group in subset.loc[subset.score_kind.eq("probability")].groupby("model"):
            observed, predicted = calibration_curve(group.hit, group.score, n_bins=8, strategy="quantile")
            ax.plot(predicted, observed, "o-", label=model)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed hit fraction", title=protocol + "\nProbability reliability")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{protocol}__reliability.png", dpi=180)
        plt.close(fig)
    chosen = selection["chosen"]
    row = results.loc[results.selected_before_test].iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int),
                           display_labels=["Non-hit candidate", "Hit"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Selected in advance: {chosen['model']}\nArtist-disjoint holdout")
    fig.tight_layout()
    fig.savefig(output / "selected_confusion_matrix.png", dpi=180)
    plt.close(fig)
