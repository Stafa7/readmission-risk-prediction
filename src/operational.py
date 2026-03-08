"""
operational.py — Phase 13: Risk Tiers, Top-K Outreach List.

Course modules:
  Module I: Data as a Business, AI Strategy
"""

import numpy as np
import pandas as pd
from .config import (
    TIER_LOW_PERCENTILE, TIER_HIGH_PERCENTILE,
    OUTREACH_EXPLANATION_N, OUTREACH_TOP_REASONS,
)


def assign_risk_tiers(best_probs):
    """Assign Low / Medium / High risk tiers based on percentile cutoffs."""
    p_lo = np.percentile(best_probs, TIER_LOW_PERCENTILE)
    p_hi = np.percentile(best_probs, TIER_HIGH_PERCENTILE)
    print(f"  Tier thresholds: Low < {p_lo:.4f} | Medium {p_lo:.4f}–{p_hi:.4f} | High > {p_hi:.4f}")

    def _tier(prob):
        if prob >= p_hi:
            return "High"
        elif prob >= p_lo:
            return "Medium"
        return "Low"

    return np.vectorize(_tier)(best_probs), p_lo, p_hi


def generate_outreach_list(explainer, X_test, best_probs, feature_names,
                           save_path=None):
    """
    Phase 13: Create a patient outreach list with risk scores,
    tiers, and SHAP-based top-3 reasons.
    """
    print("=" * 60)
    print("  PHASE 13: OPERATIONAL OUTPUTS (RISK TIERS + TOP-K)")
    print("=" * 60)

    tiers, _, _ = assign_risk_tiers(best_probs)
    probs_series = pd.Series(best_probs, index=X_test.index)

    n = min(OUTREACH_EXPLANATION_N, len(X_test))
    sample = X_test.head(n)
    print(f"  Generating per-patient SHAP explanations (top {n})...")
    sv = explainer.shap_values(sample)

    def _top_reasons(shap_row, top_n=OUTREACH_TOP_REASONS):
        top_idx = np.abs(shap_row).argsort()[-top_n:][::-1]
        return " | ".join(
            f"{feature_names[i]} ({'↑' if shap_row[i] > 0 else '↓'})"
            for i in top_idx
        )

    rows = []
    for i in range(n):
        idx = sample.index[i]
        prob = probs_series.loc[idx]
        rows.append({
            "risk_probability": round(prob, 4),
            "risk_tier": tiers[X_test.index.get_loc(idx)],
            "top_3_reasons": _top_reasons(sv[i], OUTREACH_TOP_REASONS),
        })

    out_df = pd.DataFrame(rows).sort_values("risk_probability", ascending=False)
    print(f"\n  --- Top 10 Highest-Risk Patients ---")
    print(out_df.head(10).to_string(index=False))

    if save_path:
        out_df.to_csv(save_path, index=False)
        print(f"  Saved outreach list → {save_path}")

    return out_df
