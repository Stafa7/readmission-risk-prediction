"""
splitting.py — Phase 5: Train/Test Split (Patient-Level Grouped).

IMPORTANT: The split is performed BEFORE feature selection / PCA
to prevent data leakage.  PCA, MI, and any fitted transformations
must be fit on training data only.

Course modules:
  Module II: Veridical Data Science, Best Practices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .config import SEED, TEST_SIZE


def separate_X_y(df: pd.DataFrame):
    """
    Separate the encoded DataFrame into features (X), target (y),
    and patient IDs for grouped splitting.
    """
    patient_ids = df["patient_nbr"]
    y = df["target"]
    X = df.drop(columns=["target", "patient_nbr"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"Features: {X.shape[1]}  |  Samples: {X.shape[0]:,}")
    print(f"Target rate: {y.mean()*100:.1f}%")
    return X, y, patient_ids


def train_test_split_grouped(X, y, patient_ids):
    """
    Patient-level grouped split so no patient appears in both
    train and test sets.  Returns X_train, X_test, y_train, y_test,
    groups_train.
    """
    print("=" * 60)
    print("  PHASE 5: TRAIN / TEST SPLIT (PATIENT-GROUPED)")
    print("=" * 60)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = patient_ids.iloc[train_idx]

    # Verify zero patient overlap
    train_patients = set(patient_ids.iloc[train_idx])
    test_patients = set(patient_ids.iloc[test_idx])
    overlap = train_patients & test_patients

    print(f"  Train: {len(X_train):,} encounters ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test:  {len(X_test):,} encounters ({y_test.mean()*100:.1f}% positive)")
    print(f"  Train patients: {len(train_patients):,}  |  Test patients: {len(test_patients):,}")
    print(f"  Patient overlap: {len(overlap)} ← must be 0")
    assert len(overlap) == 0, "DATA LEAKAGE: patients appear in both splits!"

    return X_train, X_test, y_train, y_test, groups_train
