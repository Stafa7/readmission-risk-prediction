"""
evaluation.py — Phases 9, 14, 15: ROC/PR Curves, Confusion Matrix, Lift.

Course modules:
  Module IV: Model Evaluation
  Module II: Enterprise DS
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix,
)


def plot_roc_pr_curves(models_probs: dict, y_test, save_path=None):
    """
    Plot ROC and Precision-Recall curves for all models.
    models_probs: {name: probs_array}
    """
    print("=" * 60)
    print("  PHASE 9: ROC & PRECISION-RECALL CURVES")
    print("=" * 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax1.plot(fpr, tpr, lw=2, label=f"{name} ({auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax2.plot(rec, prec, lw=2, label=f"{name} ({ap:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curves", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    ax2.axhline(y=y_test.mean(), color="k", ls="--", alpha=0.3, label="Baseline")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, best_probs, model_name, save_path=None):
    """
    Phase 14: Confusion matrix at the top-10 % threshold.
    """
    print("=" * 60)
    print("  PHASE 14: CONFUSION MATRIX & CLASSIFICATION REPORT")
    print("=" * 60)

    threshold = np.percentile(best_probs, 90)
    y_pred = (best_probs >= threshold).astype(int)

    print(f"  Threshold (top 10%): {threshold:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Readmitted", "Readmitted"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name} (Top 10 % Threshold)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gains_and_lift(y_test, best_probs, model_name, save_path=None):
    """
    Phase 15: Cumulative Gains and Lift charts.
    """
    print("=" * 60)
    print("  PHASE 15: CUMULATIVE GAINS & LIFT CHART")
    print("=" * 60)

    sorted_idx = np.argsort(-best_probs)
    sorted_y = y_test.values[sorted_idx]
    cumulative_gains = np.cumsum(sorted_y) / sorted_y.sum()
    pct_pop = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gains
    ax1.plot(pct_pop, cumulative_gains, color="#e74c3c", lw=2, label=model_name)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax1.set_xlabel("% of Population Contacted")
    ax1.set_ylabel("% of Readmissions Captured")
    ax1.set_title("Cumulative Gains Chart"); ax1.legend()
    ax1.axvline(x=0.10, color="gray", ls=":", alpha=0.5)
    gains_10 = cumulative_gains[int(0.10 * len(cumulative_gains)) - 1]
    ax1.annotate(f"At 10%: {gains_10*100:.1f}% captured",
                 xy=(0.10, gains_10), xytext=(0.25, 0.3), fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    # Lift
    lift = cumulative_gains / pct_pop
    ax2.plot(pct_pop, lift, color="#3498db", lw=2, label=model_name)
    ax2.axhline(y=1, color="k", ls="--", alpha=0.3, label="Random (lift=1)")
    ax2.set_xlabel("% of Population Contacted"); ax2.set_ylabel("Lift")
    ax2.set_title("Lift Chart"); ax2.legend(); ax2.set_xlim(0, 0.5)

    plt.suptitle("Operational Value — Model Prioritization",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
