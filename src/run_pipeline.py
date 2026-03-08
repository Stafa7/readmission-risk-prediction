"""
run_pipeline.py — Execute the full end-to-end ML pipeline from the command line.

Usage:
    python -m src.run_pipeline
    python -m src.run_pipeline --data path/to/diabetic_data.csv
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_FILE, OUTPUT_DIR
from src.data_cleaning import run_cleaning
from src.feature_engineering import run_feature_engineering
from src.splitting import separate_X_y, train_test_split_grouped
from src.feature_selection import run_feature_selection
from src.modeling import (
    heuristic_baseline, logistic_baseline,
    run_cv_all_models, train_final_models, compare_models,
)
from src.calibration import calibrate_model
from src.explainability import compute_shap_values, shap_summary_plots, shap_waterfall_examples
from src.fairness import audit_fairness_by_race
from src.evaluation import plot_roc_pr_curves, plot_confusion_matrix, plot_gains_and_lift
from src.operational import generate_outreach_list


def main(data_path=None):
    data_path = data_path or DATA_FILE
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Phases 1-2: Data Cleaning ────────────────────────────────
    df = run_cleaning(data_path)

    # ── Phase 3: Feature Engineering ─────────────────────────────
    df = run_feature_engineering(df)

    # ── Phase 5: Train/Test Split (BEFORE feature selection) ─────
    X, y, patient_ids = separate_X_y(df)
    X_train, X_test, y_train, y_test, groups_train = train_test_split_grouped(X, y, patient_ids)

    # ── Phase 4: Feature Selection (AFTER split — no leakage) ────
    X_train, X_test, mi_df, feature_names = run_feature_selection(
        X_train, X_test, y_train, output_dir=OUTPUT_DIR
    )

    # ── Phase 6: Baselines ───────────────────────────────────────
    print("=" * 60)
    print("  PHASE 6: BASELINE MODELS")
    print("=" * 60)
    heuristic_score = heuristic_baseline(X_test, y_test)
    lr_baseline, lr_scaler, lr_probs = logistic_baseline(X_train, X_test, y_train, y_test)

    # ── Phase 7: SOTA CV ─────────────────────────────────────────
    cv_results, scale_pos = run_cv_all_models(X_train, y_train, groups_train)

    # ── Phase 8: Final Models ────────────────────────────────────
    models, scaler = train_final_models(X_train, X_test, y_train, y_test, scale_pos)
    results_df, best_name = compare_models(models, heuristic_score, y_test)

    best_model, best_probs = models[best_name]

    # ── Phase 9: ROC & PR Curves ─────────────────────────────────
    model_probs = {name: probs for name, (_, probs) in models.items()}
    plot_roc_pr_curves(model_probs, y_test,
                       save_path=os.path.join(OUTPUT_DIR, "roc_pr_curves.png"))

    # ── Phase 10: Calibration ────────────────────────────────────
    cal_model = calibrate_model(
        best_model, best_probs, X_test, y_test, best_name,
        save_path=os.path.join(OUTPUT_DIR, "calibration.png"),
    )

    # ── Phase 11: SHAP Explainability ────────────────────────────
    shap_model = best_model
    explainer, shap_values, X_shap_sample = compute_shap_values(
        shap_model, X_test, feature_names
    )
    shap_summary_plots(shap_values, X_shap_sample, save_dir=OUTPUT_DIR)
    shap_waterfall_examples(explainer, X_test, best_probs, feature_names)

    # ── Phase 12: Fairness Audit ─────────────────────────────────
    fairness_df = audit_fairness_by_race(
        X_test, y_test, best_probs,
        save_path=os.path.join(OUTPUT_DIR, "fairness_audit.png"),
    )

    # ── Phase 13: Operational Outputs ────────────────────────────
    outreach_df = generate_outreach_list(
        explainer, X_test, best_probs, feature_names,
        save_path=os.path.join(OUTPUT_DIR, "outreach_list.csv"),
    )

    # ── Phase 14: Confusion Matrix ───────────────────────────────
    plot_confusion_matrix(y_test, best_probs, best_name,
                          save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # ── Phase 15: Gains & Lift ───────────────────────────────────
    plot_gains_and_lift(y_test, best_probs, best_name,
                        save_path=os.path.join(OUTPUT_DIR, "gains_lift.png"))

    # ── Final Summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  BEST MODEL: {best_name}")
    print(results_df.to_string(index=False))
    print("\n  PIPELINE COMPLETE ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to diabetic_data.csv")
    args = parser.parse_args()
    main(args.data)
