"""
feature_selection.py — Phase 4: Feature Selection & Dimensionality Reduction.

*** DATA-LEAKAGE FIX ***
All fitted transformations (PCA, StandardScaler for PCA) are
fit on X_train only, then applied to X_test.  Mutual Information
is also computed on X_train only.

Course modules:
  Module III: Feature Subset Selection & Dimensionality Reduction
              (Mutual Information, PCA, zero-variance removal)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from .config import SEED


def remove_zero_variance(X_train, X_test):
    """Drop features with zero variance in the training set."""
    zero_var = X_train.columns[X_train.std() == 0]
    if len(zero_var) > 0:
        X_train = X_train.drop(columns=zero_var)
        X_test = X_test.drop(columns=zero_var)
        print(f"  Removed {len(zero_var)} zero-variance features")
    else:
        print("  No zero-variance features found")
    return X_train, X_test


def compute_mutual_information(X_train, y_train, top_n: int = 20):
    """
    Compute Mutual Information scores on training data only.
    Returns a sorted DataFrame.
    """
    print("  Computing Mutual Information scores (training data only)...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=SEED, n_neighbors=5)
    mi_df = (
        pd.DataFrame({"feature": X_train.columns, "mi_score": mi_scores})
        .sort_values("mi_score", ascending=False)
    )
    print(f"\n  Top {top_n} Features by Mutual Information:")
    print(mi_df.head(top_n).to_string(index=False))
    return mi_df


def pca_visualisation(X_train, y_train, save_path=None):
    """
    Fit PCA on training data only (2-component projection for EDA).
    This is purely for exploration — no PCA features are added
    to the modelling pipeline.

    LEAKAGE FIX: scaler and PCA are fit on X_train, not on full X.
    """
    print("  PCA 2-D visualisation (fit on training data only)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_train.values, cmap="RdYlGn_r", alpha=0.1, s=5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA Projection (2-D) — Training Data, Colored by Readmission",
                 fontsize=13, fontweight="bold")
    plt.colorbar(scatter, label="Readmitted <30d")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return pca


def run_feature_selection(X_train, X_test, y_train, output_dir=None):
    """
    Execute feature selection (Phase 4) — after the train/test split.
    Returns cleaned X_train, X_test, and the MI DataFrame.
    """
    print("=" * 60)
    print("  PHASE 4: FEATURE SELECTION & DIMENSIONALITY REDUCTION")
    print("=" * 60)

    X_train, X_test = remove_zero_variance(X_train, X_test)
    mi_df = compute_mutual_information(X_train, y_train)

    pca_path = None
    if output_dir:
        import os
        pca_path = os.path.join(output_dir, "pca_2d_train.png")
    pca_visualisation(X_train, y_train, save_path=pca_path)

    feature_names = X_train.columns.tolist()
    print(f"\n  Final feature set: {len(feature_names)} features")
    return X_train, X_test, mi_df, feature_names
