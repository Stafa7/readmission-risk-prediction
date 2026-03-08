# Readmission Risk Prediction

**Predicting 30-Day Hospital Readmission for Diabetic Patients**

INSY 662 · Enterprise Data Science · Group Project

## Team
- Feroz Khan
- Matthieu Lafont
- Henry Wolcott
- Mustafa Yousaf

## Problem Statement

Hospital readmissions within 30 days cost the U.S. healthcare system over **$26B annually**. Care-management resources (transition-of-care nurses, social workers, follow-up scheduling) are finite. This project builds an end-to-end ML pipeline that predicts 30-day readmission risk for diabetic patients, producing calibrated risk scores and actionable outreach lists.

**Business Goal:** Reduce avoidable 30-day readmissions by prioritizing follow-up resources toward highest-risk patients.

**DS Goal:** Build a PoV pipeline with measurable lift over a heuristic baseline, producing explainable, calibrated risk scores.

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**
- 101,766 encounters across 130 U.S. hospitals
- 50+ raw features (demographics, utilization, clinical proxies, medications)
- Binary target: readmitted within 30 days (~11% positive rate)

## Project Structure

```
readmission-risk-prediction/
├── README.md
├── requirements.txt
├── environment.yml
├── data/                         # Place diabetic_data.csv here
├── src/
│   ├── __init__.py
│   ├── config.py                 # Hyperparameters, constants, paths
│   ├── data_cleaning.py          # Phase 1-2: Load, clean, leakage prevention
│   ├── feature_engineering.py    # Phase 3: ICD grouping, meds, encoding
│   ├── feature_selection.py      # Phase 4: MI, zero-variance, PCA (post-split)
│   ├── splitting.py              # Phase 5: Patient-level grouped train/test
│   ├── modeling.py               # Phase 6-8: Baselines, CV, SOTA models, stacking
│   ├── calibration.py            # Phase 10: Probability calibration
│   ├── explainability.py         # Phase 11: SHAP global + per-patient
│   ├── fairness.py               # Phase 12: Subgroup fairness audit
│   ├── evaluation.py             # Phase 9, 14-15: Curves, confusion, lift
│   └── operational.py            # Phase 13: Risk tiers, top-K outreach list
├── notebooks/
│   └── pipeline.ipynb            # End-to-end orchestration notebook
└── outputs/                      # Generated plots, CSVs, model artifacts
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Stafa7/readmission-risk-prediction.git
cd readmission-risk-prediction

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate readmission
```

### 2. Add Data

Download `diabetic_data.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) and place it in the `data/` directory.

### 3. Run the Pipeline

```bash
# Option A: Run the notebook
jupyter notebook notebooks/pipeline.ipynb

# Option B: Run as script
python -m src.run_pipeline
```

## Pipeline Phases & Course Module Mapping

| Phase | Description | Course Module |
|-------|-------------|---------------|
| 1 | Data Loading & EDA | Module II: Enterprise DS Lifecycle |
| 2 | Data Cleaning & Leakage Prevention | Module II: Best Practices |
| 3 | Feature Engineering (ICD-9 grouping, meds, derived) | Module III: Feature Engineering |
| 4 | Feature Selection (MI, PCA — **post-split only**) | Module III: Dimensionality Reduction |
| 5 | Train/Test Split (patient-level grouped) | Module II: Veridical DS |
| 6 | Heuristic + Logistic Regression Baselines | Module IV: Model Evaluation |
| 7 | SOTA Models: RF, XGBoost, LightGBM, Stacking (5-fold CV) | Module IV: SOTA Classification |
| 8 | Final Model Training & Test Evaluation | Module IV: Ensemble Learning |
| 9 | ROC & Precision-Recall Curves | Module IV: Model Evaluation |
| 10 | Probability Calibration (Isotonic) | Module II: Best Practices |
| 11 | Explainability (SHAP global + per-patient) | Module I: AI Strategy |
| 12 | Fairness Audit (race subgroups) | Module I: AI Strategy / Responsible AI |
| 13 | Operational Outputs (risk tiers, outreach list) | Module I: Data as a Business |
| 14 | Confusion Matrix & Classification Report | Module IV: Model Evaluation |
| 15 | Cumulative Gains & Lift Chart | Module II: Enterprise DS |

## Key Design Decisions

1. **No data leakage:** PCA and all feature transformations are fitted on training data only, then applied to test data. Train/test split uses `GroupShuffleSplit` to prevent patient overlap.
2. **PR-AUC as primary metric:** With ~11% positive rate, PR-AUC is more informative than ROC-AUC.
3. **Domain-driven feature engineering:** ICD-9 codes grouped into 18 clinical categories; medications aggregated rather than one-hot encoded.
4. **Calibration:** Isotonic regression ensures risk scores are meaningful probabilities.
5. **Fairness:** ROC-AUC and PR-AUC audited across racial subgroups with flagging for >5pp gaps.

## Results

The final LightGBM model achieves ~2.6× lift at the top decile, meaning contacting the top 10% highest-risk patients captures ~26% of all readmissions — significantly outperforming both random selection and a heuristic baseline.

## License

This project was developed for academic purposes as part of INSY 662 at McGill University.
