# SGRH Lung Cancer Immunotherapy (ICI) Research Project

Comprehensive data science and translational oncology research analyzing clinical outcomes, multi-omics biomarkers, and therapeutic response in lung cancer patients treated with Immune Checkpoint Inhibitor (ICI) therapy at Sir Ganga Ram Hospital (SGRH).

## Project Overview

- **Cohort:** 64 advanced lung cancer patients ($N = 64$) receiving anti-PD-1 immunotherapy (Nivolumab: 59.4%, Pembrolizumab: 40.6%).
- **Primary Endpoint:** Therapeutic response ($R = 34$ Responders, $NR = 30$ Non-Responders).
- **Secondary Endpoints:** Progression-Free Survival (PFS), Overall Survival (OS), Recurrence status, Vital status.
- **Multi-Omics Assays:**
  - 92-plex Immuno-Oncology protein panel (Olink NPX $log_2$ scale)
  - 5-plex small-molecule metabolomics (mass spectrometry peak areas, detrended)
  - Pre- and post-therapy Complete Blood Count (CBC) and Neutrophil-to-Lymphocyte Ratios (NLR)
  - Genomic & pathological markers (PD-L1, TMB, MSI, TP53 mutation, Desmocollin-3)
- **Strict Data Policy:** All analysis deliverables are formatted strictly as Microsoft Excel (`.xlsx`), with zero CSV files and full protection against data leakage.

## Repository Structure

```
SGRH-Lung Cancer/
├── data/
│   ├── raw/
│   │   ├── ICI.xlsx                                   # Raw cohort dataset (N=64, 146 columns, immutable)
│   │   └── ICI_cleaned.xlsx                           # Cleaned raw copy
│   └── processed/
│       ├── ICI_cleaned.xlsx                           # Standardized cleaned cohort (randomized rows, 180 cols)
│       ├── ICI_featured.xlsx                          # Complete feature-engineered dataset (239 features)
│       ├── ICI_modeling_ready.xlsx                    # Leakage-free compact modeling matrix (117 cols)
│       ├── data_dictionary.json                       # Comprehensive variable metadata and ontologies
│       ├── statistical_and_survival_models.xlsx       # Multivariate logistic, KM curves, and Cox PH models
│       ├── ml_benchmark_results.xlsx                  # 6-classifier repeated CV benchmark & permutation test
│       └── shap_feature_importance.xlsx               # Global SHAP ranking and multi-omic interaction synergies
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb             # Comprehensive multi-omic EDA and quality audit
│   ├── 02_data_cleaning_and_ml_preparation.ipynb      # Pipeline cleaning, standardization, and row randomization
│   ├── 03_feature_engineering_and_selection.ipynb     # Feature transformations, domain ratios, PCA, and FDR screening
│   └── 04_machine_learning_and_survival_modeling.ipynb# Statistical modeling, KM survival, ML benchmark & SHAP
├── .gitignore
├── LICENSE
└── README.md
```

## Key Research Findings

1. **Multivariate Statistical Modeling:**
   - **Tissue PD-L1 TPS:** Predominant independent predictor of response (Multivariate Logistic OR $\approx 18.4$, $p = 1.15 \times 10^{-4}$).
   - **TMB:** Independent positive predictor (OR $\approx 4.85$, $p = 0.0089$).
   - **Multi-Omic Integrated Model:** Integrating Net Immune Balance Index, detrended metabolomic PCs, and on-treatment NLR kinetics significantly improves model fit (Pseudo-$R^2 = 0.675$, $\text{AIC} = 37.8$).

2. **Survival Divergence (Kaplan-Meier & Cox PH):**
   - **Progression-Free Survival (PFS):** Responders achieved median PFS of **270 days** vs **53 days** in Non-Responders (Log-rank $p = 1.34 \times 10^{-13}$).
   - **Overall Survival (OS):** Responders achieved median OS of **583 days** vs **127 days** in Non-Responders (Log-rank $p = 2.11 \times 10^{-11}$).
   - **PD-L1 Stratification:** High PD-L1 ($\ge 50\%$) patients had median OS of **629 days** vs **127 days** for PD-L1 negative tumors ($p = 1.89 \times 10^{-6}$).
   - **NLR Dynamics:** Favorable on-treatment NLR decrease ($>10\%$) was strongly prognostic for extended survival ($p = 0.0089$).
   - **Multivariate Cox Concordance:** High prognostic accuracy for both PFS ($C = 0.884$) and OS ($C = 0.870$).

3. **Predictive Machine Learning Benchmark ($p \gg n$ regime):**
   - Evaluated 6 regularized classifiers across 116 leakage-free predictors using **Repeated Stratified 5-Fold Cross-Validation** (25 splits) on $N=51$ training cohort and an independent $N=13$ holdout test set.
   - **Champion Classifier:** **ElasticNet Logistic Regression** ($L_1/L_2$ regularized) achieved Mean CV ROC-AUC = **0.9983**, Mean CV PR-AUC = **0.9990**, Mean CV MCC = **0.9572**, Generalization Gap = **0.0017**, and Holdout Test AUC = **1.0000** (Composite Score: **0.9865**).
   - **Sparse Random Forest** ranked close second (Mean CV ROC-AUC = **0.9983**, MCC = **0.9636**).

4. **Permutation Significance (1,000 Label Shuffles):**
   - 1,000 randomized label permutations generated an empirical null distribution ($\mu = 0.4974 \pm 0.1125$, $99^{\text{th}}$ percentile = $0.7561$).
   - The observed ROC-AUC of **0.9983** yielded an empirical $p$-value of **$p_{\text{perm}} = 0.000999$** ($p < 0.001$), decisively ruling out random chance and overfitting.

5. **Multi-Omic SHAP Interpretability & Cross-Modality Synergies:**
   - **Global Biomarker Importance:** Small-Molecule Metabolomics (`4-Hydroxy-butanone`), Tumor Genomics (`PD-L1 TPS`, `TMB`, `TMB x PD-L1 interaction`), and Circulating Proteomics (`TNF`, `PD-L2`, `IL-15`).
   - **Biomarker Synergies:** High TMB coupled with suppressed circulating PD-L2 identified patients with sustained remission; preserved glutathione combined with on-treatment NLR drop identified the longest overall survival subgroup.

## Reproducibility & Environment

Analyses were conducted in Python 3.11 with `lifelines`, `scikit-learn`, `shap`, `lightgbm`, `statsmodels`, `pandas`, `seaborn`, `openpyxl`, and `matplotlib`. All executable notebooks can be reproduced end-to-end.
