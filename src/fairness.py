"""
fairness.py — Phase 12: Fairness Audit across demographic subgroups.

Course modules:
  Module I: AI Strategy, Responsible AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from .config import FAIRNESS_MIN_N, FAIRNESS_MIN_POS, FAIRNESS_GAP_THRESHOLD


def audit_fairness_by_race(X_test, y_test, best_probs, save_path=None):
    """
    Evaluate model performance across racial subgroups.
    Flags any subgroup with ROC-AUC gap > threshold from population.
    """
    print("=" * 60)
    print("  PHASE 12: FAIRNESS AUDIT")
    print("=" * 60)

    # Reconstruct race from one-hot columns
    race_cols = [c for c in X_test.columns if c.startswith("race_")]
    test_race = pd.DataFrame(X_test[race_cols])
    test_race["race"] = "Caucasian"  # reference (dropped) category
    for col in race_cols:
        test_race.loc[test_race[col] == 1, "race"] = col.replace("race_", "")
    test_race["y_true"] = y_test.values
    test_race["y_prob"] = best_probs

    header = f"  {'Race Group':<20} {'N':>6} {'Base Rate':>10} {'ROC-AUC':>9} {'PR-AUC':>8} {'Recall@10%':>11}"
    print(header)
    print("  " + "-" * 68)

    fairness_results = []
    for race in sorted(test_race["race"].unique()):
        subset = test_race[test_race["race"] == race]
        n = len(subset)
        if n < FAIRNESS_MIN_N or subset["y_true"].sum() < FAIRNESS_MIN_POS:
            continue
        br = subset["y_true"].mean()
        ra = roc_auc_score(subset["y_true"], subset["y_prob"])
        pa = average_precision_score(subset["y_true"], subset["y_prob"])
        k_sub = max(1, int(n * 0.10))
        topk = subset.nlargest(k_sub, "y_prob")
        rec_k = topk["y_true"].sum() / max(1, subset["y_true"].sum())
        print(f"  {race:<20} {n:>6} {br:>10.3f} {ra:>9.4f} {pa:>8.4f} {rec_k:>11.4f}")
        fairness_results.append({"race": race, "n": n, "roc_auc": ra, "pr_auc": pa})

    pop_roc = roc_auc_score(y_test, best_probs)
    print(f"\n  Population ROC-AUC: {pop_roc:.4f}")

    fairness_df = pd.DataFrame(fairness_results)
    gap = fairness_df["roc_auc"].min() - pop_roc
    flag = "⚠ FLAG" if abs(gap) > FAIRNESS_GAP_THRESHOLD else "✓ OK"
    print(f"  Max subgroup gap: {gap:.4f} {flag}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(fairness_df)), fairness_df["roc_auc"],
           color="#3498db", alpha=0.8)
    ax.axhline(y=pop_roc, color="red", ls="--", label=f"Population ({pop_roc:.3f})")
    ax.set_xticks(range(len(fairness_df)))
    ax.set_xticklabels(fairness_df["race"], rotation=30, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Fairness Audit — ROC-AUC by Race", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fairness_df
