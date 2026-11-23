# SGRH Lung Cancer Immunotherapy (ICI) Research Project

Comprehensive data science and translational oncology research analyzing clinical outcomes, multi-omics biomarkers, and therapeutic response in lung cancer patients treated with Immune Checkpoint Inhibitor (ICI) therapy at Sir Ganga Ram Hospital (SGRH).

## Project Overview

- **Cohort:** 64 advanced lung cancer patients ($N = 64$) receiving anti-PD-1 immunotherapy (Nivolumab: 59.4%, Pembrolizumab: 40.6%).
- **Primary Endpoint:** Therapeutic response ($R = 34$ Responders, $NR = 30$ Non-Responders).
- **Secondary Endpoints:** Progression-Free Survival (PFS), Overall Survival (OS), Recurrence status, Vital status.
- **Multi-Omics Assays:**
  - 92-plex Immuno-Oncology protein panel (Olink NPX $log_2$ scale)
  - 5-plex small-molecule metabolomics (mass spectrometry peak areas)
  - Pre- and post-therapy Complete Blood Count (CBC) and Neutrophil-to-Lymphocyte Ratios (NLR)
  - Genomic & pathological markers (PD-L1, TMB, MSI, TP53 mutation, Desmocollin-3)

## Repository Structure

```
SGRH-Lung Cancer/
├── data/
│   └── raw/
│       └── ICI.xlsx                               # Raw cohort dataset (N=64, 146 columns)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb         # Fully executed end-to-end data science EDA
│   └── generate_notebook.py                       # Programmatic reproducible notebook generator
├── knowledge_graph/
│   ├── kg_data.json                               # Structured JSON Knowledge Graph (entities, ontologies, relations)
│   └── generate_kg.py                             # Generator script for Knowledge Graph
├── backups/
│   ├── README.md                                  # Immutable backup policy and index
│   ├── backup_manifest.json                       # Version registry
│   ├── v001_initial/                              # Baseline SHA-256 verified raw snapshot
│   └── v002_eda_and_knowledge_graph/              # Snapshot of analysis notebook & KG
├── .gitignore
└── README.md
```

## Key Findings & Data Quality Audit

1. **Top Predictive Biomarkers:**
   - **PD-L1 Expression:** Trimodal ($0\%, 30\%, 60\%$), highly significant association with response ($p = 6.27 \times 10^{-12}$, FDR $q = 6.33 \times 10^{-10}$).
   - **Tumor Mutational Burden (TMB):** Elevated in Responders (median 13.5 vs 3.15 mut/Mb, $p = 6.95 \times 10^{-6}$, FDR $q = 2.34 \times 10^{-4}$).
   - **Proteins:** PD-L2, DCN, IL15, CCL3, TWEAK, IL10 significantly downregulated in Responders ($q < 0.05$).
2. **Critical Data Quality Issues Identified:**
   - **$PFS > OS$ Inversion:** Identified in 4 patients (Pts 1, 21, 28, 53) requiring clinical chart verification.
   - **Run-Order / Batch Confounding:** Monotonic instrument drift in metabolomics and proteomics strongly confounded with outcome-sorted spreadsheet layout.
   - **Data Entry & Survey Issues:** Trailing whitespace in Patient 10 `sex == 'F '`, typo in Patient 40 `non-veg. == '1\``, 8 patients with BMI discrepancy $> 1.0\text{ kg/m}^2$, and 31 patients recorded as both vegetarian and non-vegetarian.
   - **High-Dimensional Risk:** $p = 144$ candidate features for $n = 64$ samples ($p/n = 2.25$).

## Reproducibility & Environment

Analyses were conducted in Python 3.11 with `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, and `seaborn`.
The primary analysis notebook [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb) can be reproduced and re-executed end-to-end.
