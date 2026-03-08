"""
modeling.py — Phases 6, 7, 8: Baseline Models, SOTA CV, Final Training.

Course modules:
  Module IV: SOTA Classification (LR, RF, XGBoost, LightGBM, Stacking,
             Ensemble Learning, Model Evaluation)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
import xgboost as xgb
import lightgbm as lgb

from .config import (
    SEED, N_FOLDS, TOP_K_PCT,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, LGB_PARAMS,
    STACK_LR, STACK_RF, STACK_XGB,
)


# ── Cross-validation helper ──────────────────────────────────────

def cross_validate_model(model, X_data, y_data, groups, model_name,
                         needs_scaling=False):
    """
    Stratified-Group K-Fold CV with PR-AUC, ROC-AUC, Brier.
    Groups ensure no patient overlap across folds.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_data, y_data, groups)):
        X_tr, X_val = X_data.iloc[tr_idx], X_data.iloc[val_idx]
        y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]

        if needs_scaling:
            sc = StandardScaler()
            X_tr = pd.DataFrame(sc.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_val = pd.DataFrame(sc.transform(X_val), columns=X_val.columns, index=X_val.index)

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]

        fold_metrics.append({
            "fold": fold + 1,
            "pr_auc": average_precision_score(y_val, probs),
            "roc_auc": roc_auc_score(y_val, probs),
            "brier": brier_score_loss(y_val, probs),
        })

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n  {model_name} — {N_FOLDS}-Fold CV:")
    print(f"    PR-AUC:  {metrics_df['pr_auc'].mean():.4f} ± {metrics_df['pr_auc'].std():.4f}")
    print(f"    ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")
    print(f"    Brier:   {metrics_df['brier'].mean():.4f} ± {metrics_df['brier'].std():.4f}")
    return metrics_df


# ── Phase 6: Baselines ───────────────────────────────────────────

def heuristic_baseline(X_test, y_test):
    """Simple heuristic: rank patients by prior inpatient + emergency visits."""
    print("\n  --- Heuristic Baseline: Prior Utilization Ranking ---")
    score = X_test["number_inpatient"] + X_test["number_emergency"]
    K = int(len(y_test) * TOP_K_PCT)

    prauc = average_precision_score(y_test, score)
    rocauc = roc_auc_score(y_test, score)
    topk_idx = score.nlargest(K).index
    recall_k = y_test.loc[topk_idx].sum() / y_test.sum()
    precision_k = y_test.loc[topk_idx].mean()
    lift_k = precision_k / y_test.mean()

    print(f"    PR-AUC:     {prauc:.4f}")
    print(f"    ROC-AUC:    {rocauc:.4f}")
    print(f"    Recall@10%: {recall_k:.4f}")
    print(f"    Lift@10%:   {lift_k:.2f}x")
    return score


def logistic_baseline(X_train, X_test, y_train, y_test):
    """Train a simple Logistic Regression baseline on scaled data."""
    print("\n  --- Logistic Regression Baseline ---")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr_s, y_train)
    probs = lr.predict_proba(X_te_s)[:, 1]

    K = int(len(y_test) * TOP_K_PCT)
    print(f"    PR-AUC:     {average_precision_score(y_test, probs):.4f}")
    print(f"    ROC-AUC:    {roc_auc_score(y_test, probs):.4f}")
    print(f"    Brier:      {brier_score_loss(y_test, probs):.4f}")
    return lr, scaler, probs


# ── Phase 7: SOTA Supervised Learning (CV) ───────────────────────

def run_cv_all_models(X_train, y_train, groups_train):
    """Run 5-fold stratified-group CV for all candidate models."""
    print("=" * 60)
    print("  PHASE 7: SOTA SUPERVISED LEARNING (5-FOLD CV)")
    print("=" * 60)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight: {scale_pos:.1f}")

    lr_cv = LogisticRegression(**LR_PARAMS)
    lr_metrics = cross_validate_model(lr_cv, X_train, y_train, groups_train,
                                      "Logistic Regression", needs_scaling=True)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf_metrics = cross_validate_model(rf, X_train, y_train, groups_train,
                                      "Random Forest")

    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos, **XGB_PARAMS)
    xgb_metrics = cross_validate_model(xgb_model, X_train, y_train, groups_train,
                                       "XGBoost")

    lgb_model = lgb.LGBMClassifier(scale_pos_weight=scale_pos, **LGB_PARAMS)
    lgb_metrics = cross_validate_model(lgb_model, X_train, y_train, groups_train,
                                       "LightGBM")

    return {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "LightGBM": lgb_metrics,
    }, scale_pos


# ── Phase 8: Final Model Training ────────────────────────────────

def train_final_models(X_train, X_test, y_train, y_test, scale_pos):
    """Train all models on the full training set, evaluate on test."""
    print("=" * 60)
    print("  PHASE 8: FINAL MODEL TRAINING & TEST EVALUATION")
    print("=" * 60)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Logistic Regression
    lr_final = LogisticRegression(**LR_PARAMS)
    lr_final.fit(X_tr_s, y_train)
    lr_probs = lr_final.predict_proba(X_te_s)[:, 1]

    # Random Forest
    rf_final = RandomForestClassifier(**RF_PARAMS)
    rf_final.fit(X_train, y_train)
    rf_probs = rf_final.predict_proba(X_test)[:, 1]

    # XGBoost
    xgb_final = xgb.XGBClassifier(scale_pos_weight=scale_pos, **XGB_PARAMS)
    xgb_final.fit(X_train, y_train)
    xgb_probs = xgb_final.predict_proba(X_test)[:, 1]

    # LightGBM
    lgb_final = lgb.LGBMClassifier(scale_pos_weight=scale_pos, **LGB_PARAMS)
    lgb_final.fit(X_train, y_train)
    lgb_probs = lgb_final.predict_proba(X_test)[:, 1]

    # Stacking Ensemble
    print("  Training Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(**STACK_LR)),
            ("rf", RandomForestClassifier(**STACK_RF)),
            ("xgb", xgb.XGBClassifier(scale_pos_weight=scale_pos, **STACK_XGB)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3, passthrough=False, n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    stacking_probs = stacking.predict_proba(X_test)[:, 1]

    models = {
        "Logistic Regression": (lr_final, lr_probs),
        "Random Forest": (rf_final, rf_probs),
        "XGBoost": (xgb_final, xgb_probs),
        "LightGBM": (lgb_final, lgb_probs),
        "Stacking": (stacking, stacking_probs),
    }
    print("  All models trained.")
    return models, scaler


# ── Model Comparison ─────────────────────────────────────────────

def evaluate_model(name, y_true, probs, K):
    """Compute comprehensive evaluation metrics for a single model."""
    probs = np.array(probs, dtype=float)
    if probs.max() > 1 or probs.min() < 0:
        probs = (probs - probs.min()) / (probs.max() - probs.min())

    prauc = average_precision_score(y_true, probs)
    rocauc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    topk_idx = pd.Series(probs, index=y_true.index).nlargest(K).index
    recall_k = y_true.loc[topk_idx].sum() / y_true.sum()
    precision_k = y_true.loc[topk_idx].mean()
    lift_k = precision_k / y_true.mean()

    return {
        "Model": name, "PR-AUC": prauc, "ROC-AUC": rocauc, "Brier": brier,
        "Recall@10%": recall_k, "Precision@10%": precision_k, "Lift@10%": lift_k,
    }


def compare_models(models_dict, heuristic_score, y_test):
    """Build a comparison DataFrame across all models + heuristic."""
    K = int(len(y_test) * TOP_K_PCT)
    results = [evaluate_model("Heuristic", y_test, heuristic_score.values.astype(float), K)]
    for name, (_, probs) in models_dict.items():
        results.append(evaluate_model(name, y_test, probs, K))

    results_df = pd.DataFrame(results).round(4)
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — TEST SET")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_name = results_df.loc[results_df["PR-AUC"].idxmax(), "Model"]
    print(f"\n  ★ Best model by PR-AUC: {best_name}")
    return results_df, best_name
