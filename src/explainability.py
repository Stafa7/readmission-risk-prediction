"""
explainability.py — Phase 11: SHAP Explainability (global + per-patient).

Course modules:
  Module I: AI Strategy (responsible AI, explainability)
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from .config import SEED, SHAP_SAMPLE_SIZE


def compute_shap_values(model, X_test, feature_names):
    """
    Compute SHAP values using TreeExplainer on a sample of test data.
    Returns explainer, shap_values, and the sample DataFrame.
    """
    print("=" * 60)
    print("  PHASE 11: EXPLAINABILITY (SHAP)")
    print("=" * 60)

    explainer = shap.TreeExplainer(model)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=SEED)
    shap_values = explainer.shap_values(X_sample)

    print(f"  SHAP computed on {sample_size} test samples")
    return explainer, shap_values, X_sample


def shap_summary_plots(shap_values, X_sample, save_dir=None):
    """Generate SHAP summary (beeswarm) and bar plots."""
    import os

    # Beeswarm
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Bar
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.show()


def shap_waterfall_examples(explainer, X_test, best_probs, feature_names):
    """Show waterfall plots for high, medium, and low risk patients."""
    probs_series = pd.Series(best_probs, index=X_test.index)

    high_idx = probs_series.idxmax()
    low_idx = probs_series.idxmin()
    med_idx = (probs_series - probs_series.median()).abs().idxmin()

    for label, idx in [("High Risk", high_idx), ("Medium Risk", med_idx), ("Low Risk", low_idx)]:
        row = X_test.loc[[idx]]
        sv = explainer.shap_values(row)
        prob = probs_series.loc[idx]
        print(f"\n  {label} Patient — P(readmit<30d) = {prob:.3f}")
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=explainer.expected_value,
                data=row.values[0],
                feature_names=feature_names,
            ),
            max_display=12, show=True,
        )
