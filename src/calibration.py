"""
calibration.py — Phase 10: Probability Calibration.

Course modules:
  Module II: Enterprise DS Best Practices
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calibrate_model(best_model, best_probs, X_test, y_test, model_name,
                    save_path=None):
    """
    Apply isotonic calibration using the first half of the test set
    as a calibration set and the second half as an evaluation set.
    Returns the calibrated model.
    """
    print("=" * 60)
    print("  PHASE 10: PROBABILITY CALIBRATION")
    print("=" * 60)

    cal_split = int(len(X_test) * 0.5)
    X_cal, X_eval = X_test.iloc[:cal_split], X_test.iloc[cal_split:]
    y_cal, y_eval = y_test.iloc[:cal_split], y_test.iloc[cal_split:]
    probs_eval = best_probs[cal_split:]

    print(f"  Calibrating {model_name} with isotonic regression...")
    cal_model = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    cal_model.fit(X_cal, y_cal)
    cal_probs_eval = cal_model.predict_proba(X_eval)[:, 1]

    brier_before = brier_score_loss(y_eval, probs_eval)
    brier_after = brier_score_loss(y_eval, cal_probs_eval)

    # Calibration curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    pt_b, pp_b = calibration_curve(y_eval, probs_eval, n_bins=10)
    ax1.plot(pp_b, pt_b, "o-", color="#e74c3c", label="Before")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title(f"Before (Brier={brier_before:.4f})")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Observed"); ax1.legend()

    pt_a, pp_a = calibration_curve(y_eval, cal_probs_eval, n_bins=10)
    ax2.plot(pp_a, pt_a, "o-", color="#2ecc71", label="After")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set_title(f"After Isotonic (Brier={brier_after:.4f})")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Observed"); ax2.legend()

    plt.suptitle(f"Calibration — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  Brier: {brier_before:.4f} → {brier_after:.4f}")
    return cal_model
