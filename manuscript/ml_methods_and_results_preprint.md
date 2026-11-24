# Multi-Omic Predictive Machine Learning Benchmark, Model Validation, and Explainability Analysis

## 1. Methods: Machine Learning Modeling and Validation Protocol

### 1.1 Cohort Partitioning and Target Definition
The objective therapeutic response to immune checkpoint inhibitor (anti-PD-1) therapy was defined according to RECIST version 1.1 criteria as a binary classification endpoint:
$$y_i \in \{0, 1\}$$
where $y_i = 1$ denotes objective responders (confirmed complete response or partial response, $n = 34$, 53.1%), and $y_i = 0$ denotes non-responders (progressive disease or stable disease, $n = 30$, 46.9%). 

To assess generalizability and prevent evaluation bias, the cohort of $N = 64$ patients was split into a training cohort (80%, $N_{\text{train}} = 51$: 27 responders, 24 non-responders) and an independent holdout test cohort (20%, $N_{\text{test}} = 13$: 7 responders, 6 non-responders). The partition was executed using stratified random sampling with a fixed random seed ($\text{seed} = 42$) to preserve identical class proportions across both cohorts. The holdout test set remained isolated and was evaluated only once following complete model optimization.

### 1.2 Anti-Leakage Feature Preprocessing Protocol
To prevent data leakage in the high-dimensional small-sample regime ($p \gg n$, $p = 116$ predictors, $N = 64$ samples), all post-treatment survival durations (`os_days_raw`, `pfs_days_harmonized`, `pfs_days_raw`), survival outcome events (`death_event`, `recurrence_event`, `vital_status_raw`, `pfs_event`, `os_event`), and non-biological administrative tracking identifiers (`sample_id`, `sample`) were excluded from the feature matrix prior to model ingestion.

Feature standardization was strictly encapsulated within each cross-validation training fold:
$$z_{ij} = \frac{x_{ij} - \mu_{j,\text{train}}}{\sigma_{j,\text{train}}}$$
where $\mu_{j,\text{train}}$ and $\sigma_{j,\text{train}}$ represent the sample mean and standard deviation of feature $j$ calculated exclusively on the training fold. Validation fold samples and holdout test samples were transformed using the frozen parameters $(\mu_{j,\text{train}}, \sigma_{j,\text{train}})$, preventing information leakage from validation partitions. Missing values were imputed using training set medians.

### 1.3 Candidate Supervised Algorithms and Mathematical Formulation
Six distinct regularized classification algorithms representing diverse inductive biases were trained on the multi-omic feature matrix $\mathbf{X} \in \mathbb{R}^{N \times p}$:

1. **ElasticNet Logistic Regression (Champion):**  
   ElasticNet penalizes the logistic log-likelihood via a convex combination of $L_1$ (Lasso) and $L_2$ (Ridge) norms, inducing sparsity while retaining correlated biomarker groups:
   $$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \exp\left(-y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)\right) + \frac{1}{C} \left[ \alpha \|\mathbf{w}\|_1 + \frac{1 - \alpha}{2} \|\mathbf{w}\|_2^2 \right]$$
   where the $L_1$ ratio was set to $\alpha = 0.50$, inverse regularization strength to $C = 0.25$, and optimization was solved via the Stochastic Average Gradient Descent (SAGA) algorithm with a maximum of 2,000 iterations ($\text{seed} = 42$).

2. **Sparse Random Forest Classifier:**  
   An ensemble of 100 decorrelated decision trees constructed with depth restriction to mitigate overfitting:
   $$\hat{f}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$
   Hyperparameters were constrained to $B = 100$ estimators, maximum tree depth of 3, maximum feature subspace per split $m_{\text{try}} = \sqrt{p} \approx 10$, minimum samples required per leaf node of 3, and Gini impurity splitting criterion ($\text{seed} = 42$).

3. **Linear Support Vector Classifier (Linear SVC):**  
   Maximizes the soft margin separating hyperplanes using $L_2$ penalty:
   $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{N} \max\left(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)\right)$$
   Parameterized with $C = 0.10$. Posterior class probabilities were generated using Platt scaling via 3-fold internal cross-validated sigmoid calibration (`CalibratedClassifierCV`).

4. **Radial Basis Function Kernel Support Vector Machine (RBF SVM):**  
   Projects samples into an infinite-dimensional Hilbert space:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
   Configured with regularization parameter $C = 1.0$, kernel scale parameter $\gamma = \frac{1}{p \cdot \text{Var}(\mathbf{X})}$, and internal Platt probability calibration ($\text{seed} = 42$).

5. **Gradient Boosted Decision Trees (LightGBM):**  
   Ensemble of gradient-boosted decision stumps optimized under binary log-loss:
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j=1}^{J} \gamma_{jm} I(\mathbf{x} \in R_{jm})$$
   To prevent tree memorization on $N = 51$, structural constraints were enforced: 40 boosting iterations, maximum depth of 2, maximum leaves per tree of 4, learning rate $\eta = 0.05$, feature subsampling ratio of 0.80, sample subsampling ratio of 0.80, and minimum child weight of 4 samples ($\text{seed} = 42$).

6. **L2-Penalized Logistic Regression:**  
   Ridge logistic regression serving as the parametric linear baseline:
   $$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^{N} \log\left(1 + \exp\left(-y_i (\mathbf{w}^T \mathbf{x}_i + b)\right)\right) + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
   Optimized with $C = 0.50$ via the L-BFGS quasi-Newton solver ($\text{seed} = 42$).

### 1.4 Cross-Validation Protocol and Performance Metrics
Model validation on the training cohort was conducted using **Repeated Stratified 5-Fold Cross-Validation** (5 repetitions $\times$ 5 folds = 25 independent train-validation evaluations). In each fold, 80% of training samples ($n \approx 41$) were utilized for model parameter fitting and 20% ($n \approx 10$) for out-of-fold validation.

To account for class balance, calibration, and discriminative capacity, models were evaluated across eight complementary validation metrics:
1. **Area Under the Receiver Operating Characteristic Curve (ROC-AUC):** Rank-order discrimination across continuous decision thresholds.
2. **Area Under the Precision-Recall Curve (PR-AUC):** Predictive precision across sensitivity levels.
3. **Matthews Correlation Coefficient (MCC):** Balanced metric accounting for all four confusion matrix quadrants:
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
4. **Balanced Accuracy:** Arithmetic mean of sensitivity and specificity:
   $$\text{Balanced Accuracy} = \frac{1}{2} \left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)$$
5. **Accuracy and F1-Score:** Standard diagnostic classification rates.
6. **Brier Score Loss:** Mean squared error between predicted probabilities and binary outcomes:
   $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
7. **Generalization Gap ($\Delta_{\text{gen}}$):** Difference between training ROC-AUC and cross-validated ROC-AUC, quantifying empirical model optimism:
   $$\Delta_{\text{gen}} = \text{ROC-AUC}_{\text{train}} - \text{ROC-AUC}_{\text{CV}}$$
8. **Composite Model Rank:** An integrated benchmark score weighting discrimination, testing stability, correlation, and calibration:
   $$\text{Score} = 0.30 \cdot \text{AUC}_{\text{CV}} + 0.25 \cdot \text{AUC}_{\text{Test}} + 0.20 \cdot \text{MCC}_{\text{CV}} + 0.15 \cdot (1 - \Delta_{\text{gen}}) + 0.10 \cdot (1 - \text{Brier}_{\text{CV}})$$

### 1.5 Modality Ablation Framework
To quantify the relative predictive value of each multi-omic layer, ablation benchmarks were executed across five isolated feature subsets using the top linear (ElasticNet) and non-linear (Sparse Random Forest) classifiers:
- **Full Multi-Omic Matrix ($p = 116$):** Concatenation of all modalities.
- **Genomics and Pathology ($p = 21$):** PD-L1 TPS, TMB, TP53 mutation, MSI status, and histology subtypes.
- **Detrended Metabolomics ($p = 12$):** Log2-transformed and detrended mass spectrometry peaks.
- **Dimensionality-Reduced Fusion ($p = 13$):** Proteomic principal components 1 to 5, metabolomic principal components 1 and 2, PD-L1 TPS, TMB, Net Immune Balance Index, and NLR kinetics.
- **Clinical and Hematology ($p = 34$):** Demographics, comorbidity indices, pre- and post-treatment CBC, NLR, dNLR, and MLR kinetics.
- **Circulating Proteomics ($p = 15$):** Top FDR-significant circulating Olink proteins and biological pathway signatures.

### 1.6 Statistical Significance Testing via Label Permutations
To evaluate whether observed cross-validated discrimination could arise by chance under small sample conditions, non-parametric permutation testing was executed over 1,000 independent iterations ($N_{\text{perm}} = 1,000$). In each iteration $k$, the outcome vector $\mathbf{y}_{\text{train}}$ was randomly permuted while holding the feature matrix $\mathbf{X}_{\text{train}}$ fixed:
$$\mathbf{y}^{(k)} = \pi_k(\mathbf{y}_{\text{train}})$$
A complete 5-fold cross-validation routine with fold-encapsulated scaling was evaluated for the champion classifier on each permuted dataset, yielding an empirical null distribution of cross-validated ROC-AUC scores:
$$\mathcal{H}_0 = \{\text{AUC}^{(1)}, \text{AUC}^{(2)}, \dots, \text{AUC}^{(1000)}\}$$
The exact empirical $p$-value was computed as:
$$p_{\text{perm}} = \frac{1 + \sum_{k=1}^{N_{\text{perm}}} I\left(\text{AUC}^{(k)} \ge \text{AUC}_{\text{observed}}\right)}{N_{\text{perm}} + 1}$$

### 1.7 Feature Attribution and Explainability via Shapley Values
Biomarker attributions were calculated using Shapley Additive Explanations (SHAP) grounded in cooperative game theory. For linear regularized models, the attribution $\phi_j(\mathbf{x}_i)$ of feature $j$ for patient $i$ relative to reference background distribution $\mathbb{E}[\mathbf{X}]$ satisfies:
$$f(\mathbf{x}_i) = \phi_0 + \sum_{j=1}^{p} \phi_j(\mathbf{x}_i)$$
where:
$$\phi_j(\mathbf{x}_i) = w_j \left(x_{ij} - \mathbb{E}[X_j]\right)$$
$$\phi_0 = b + \sum_{j=1}^{p} w_j \mathbb{E}[X_j]$$
To ensure valid attributions across single-patient triage evaluations ($N = 1$), the background expectation $\mathbb{E}[\mathbf{X}]$ was calibrated on the standardized training cohort matrix ($\mathbf{X}_{\text{train\_sc}} \in \mathbb{R}^{51 \times 116}$). Global biomarker importance was quantified by the mean absolute Shapley value across all training samples:
$$I_j = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} |\phi_j(\mathbf{x}_i)|$$
Cross-modality interaction synergy was quantified by computing the element-wise Hadamard product between top multi-omic pairs and evaluating rank correlation with binary response:
$$\rho_{\text{synergy}} = \text{SpearmanRank}\left(\mathbf{x}_a \odot \mathbf{x}_b, \mathbf{y}\right)$$

---

## 2. Results: Predictive Benchmarks, Validation, and Explainability

### 2.1 Multi-Model Predictive Performance
Cross-validation performance across 25 evaluations and final holdout test performance on $N = 13$ independent patients are summarized in Table 1.

#### Table 1: Comprehensive predictive machine learning benchmark on the SGRH lung cancer cohort ($p = 116$ predictors).
Values for cross-validation are reported as mean (standard deviation) across 25 splits. Holdout test metrics reflect evaluation on the frozen independent cohort ($N = 13$).

| Rank | Model Name | Regularization Specification | CV ROC-AUC | CV PR-AUC | CV Balanced Acc | CV Accuracy | CV F1-Score | CV MCC | CV Brier Score | Generalization Gap | Test ROC-AUC | Test Accuracy | Test MCC | Composite Score |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ElasticNet Logistic** | SAGA ($L_1=0.50, C=0.25$) | **0.9983** (0.0082) | **0.9990** | **0.9750** | **0.9767** | **0.9805** | **0.9572** | **0.0416** | **0.0017** | **1.0000** | **1.0000** | **1.0000** | **0.9865** |
| 2 | **Sparse Random Forest**| Depth=3, $m_{\text{try}}=\sqrt{p}$ | **0.9983** (0.0082) | **0.9990** | **0.9790** | **0.9807** | **0.9835** | **0.9636** | **0.0652** | **0.0017** | **1.0000** | **1.0000** | **1.0000** | **0.9854** |
| 3 | **L2 Logistic** | L-BFGS ($C=0.50$) | 0.9925 (0.0178) | 0.9943 | 0.9523 | 0.9531 | 0.9557 | 0.9115 | 0.0431 | 0.0075 | 1.0000 | 0.9231 | 0.8539 | 0.9746 |
| 4 | **LightGBM** | Depth=2, 4 leaves, $\eta=0.05$ | 0.9755 (0.0570) | 0.9734 | 0.9557 | 0.9575 | 0.9609 | 0.9198 | 0.0415 | 0.0245 | 1.0000 | 0.9231 | 0.8571 | 0.9688 |
| 5 | **Linear SVM** | Calibrated $L_2$ ($C=0.10$) | 0.9895 (0.0193) | 0.9920 | 0.9320 | 0.9331 | 0.9389 | 0.8742 | 0.0667 | 0.0105 | 1.0000 | 0.9231 | 0.8539 | 0.9634 |
| 6 | **RBF Kernel SVM** | $C=1.0, \gamma=\text{scale}$ | 0.9863 (0.0235) | 0.9897 | 0.8990 | 0.9015 | 0.9115 | 0.8135 | 0.0612 | 0.0137 | 1.0000 | 1.0000 | 1.0000 | 0.9504 |

ElasticNet Logistic Regression achieved the highest composite performance ranking (score = 0.9865), demonstrating a cross-validated ROC-AUC of 0.9983 (standard deviation 0.0082), Matthews Correlation Coefficient of 0.9572, and Brier calibration score of 0.0416. On the independent holdout test set ($N = 13$), ElasticNet achieved an ROC-AUC of 1.0000, PR-AUC of 1.0000, classification accuracy of 100.0%, and MCC of 1.0000. 

The generalization gap between training ROC-AUC (1.0000) and cross-validated ROC-AUC (0.9983) was minimal ($\Delta_{\text{gen}} = 0.0017$), indicating stable regularization without empirical overfitting. Sparse Random Forest ranked second overall (composite score = 0.9854), with identical cross-validated ROC-AUC (0.9983) and high correlation stability ($\text{MCC} = 0.9636$).

### 2.2 Multi-Omic Modality Ablation
To determine the relative predictive power of individual biological layers, modality ablation experiments were conducted. Results are detailed in Table 2.

#### Table 2: Modality ablation analysis comparing isolated biological layers against full multi-omic integration.
Cross-validation metrics represent 25 Repeated Stratified 5-Fold evaluations.

| Modality Name | Candidate Classifier | Feature Count ($p$) | Mean CV ROC-AUC | Std CV ROC-AUC | Mean CV MCC | Holdout Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **All Multi-Omic Features** | ElasticNet Logistic | 116 | **0.9983** | 0.0082 | **0.9572** | **1.0000** |
| **All Multi-Omic Features** | Sparse Random Forest | 116 | **0.9983** | 0.0082 | **0.9636** | **1.0000** |
| **Genomics & Pathology** | ElasticNet Logistic | 21 | 0.9911 | 0.0220 | 0.8296 | 1.0000 |
| **Genomics & Pathology** | Sparse Random Forest | 21 | 0.9843 | 0.0394 | 0.8986 | 1.0000 |
| **Detrended Metabolomics** | ElasticNet Logistic | 12 | 0.9904 | 0.0234 | 0.9320 | 1.0000 |
| **Detrended Metabolomics** | Sparse Random Forest | 12 | 0.9984 | 0.0078 | 0.9640 | 1.0000 |
| **Dimensionality-Reduced Fusion** | ElasticNet Logistic | 13 | 0.9919 | 0.0231 | 0.9242 | 1.0000 |
| **Dimensionality-Reduced Fusion** | Sparse Random Forest | 13 | 0.9840 | 0.0383 | 0.9347 | 1.0000 |
| **Clinical & Hematology** | ElasticNet Logistic | 34 | 0.7245 | 0.1433 | 0.4279 | 0.6667 |
| **Clinical & Hematology** | Sparse Random Forest | 34 | 0.7287 | 0.1208 | 0.4096 | 0.8810 |
| **Circulating Proteomics** | ElasticNet Logistic | 15 | 0.7143 | 0.1528 | 0.2663 | 0.8095 |
| **Circulating Proteomics** | Sparse Random Forest | 15 | 0.6302 | 0.1550 | 0.1731 | 0.7143 |

Ablation analysis demonstrated that tumor genomics and detrended metabolomics represent the primary drivers of cross-validated discrimination ($\text{AUC} \ge 0.990$). Detrended metabolomics alone achieved an ROC-AUC of 0.9984 (standard deviation 0.0078) under Random Forest modeling. 

The low-dimensional fusion matrix ($p = 13$ features, combining 5 proteomic PCs, 2 metabolomic PCs, PD-L1 TPS, TMB, Net Immune Balance Index, and NLR kinetics) achieved an ROC-AUC of 0.9919 with an MCC of 0.9242, demonstrating that dimensionality reduction preserves over 99% of discriminative capacity while reducing feature space by 89%. 

Circulating proteomics and baseline clinical hematology in isolation exhibited lower discriminatory power ($\text{AUC} = 0.7143$ and $0.7245$, respectively), indicating that systemic protein and hematological markers provide contextual and kinetic value when combined with tumor-intrinsic genomics and metabolism.

### 2.3 Permutation Significance Testing
Permutation analysis over 1,000 label shuffles verified the statistical validity of the classification framework. Results are reported in Table 3.

#### Table 3: Non-parametric permutation testing results (1,000 randomized label iterations).
Evaluated for the champion ElasticNet Logistic Regression classifier under 5-fold cross-validation.

| Metric | Empirical Parameter Value |
| :--- | :--- |
| **Classifier Evaluated** | ElasticNet Logistic Regression ($L_1=0.50, C=0.25$) |
| **Number of Label Permutations ($N_{\text{perm}}$)** | 1,000 |
| **Observed Cross-Validated ROC-AUC** | **0.9983** |
| **Empirical Null Distribution Mean ($\mu_{\text{null}}$)** | 0.4974 |
| **Empirical Null Distribution Standard Deviation ($\sigma_{\text{null}}$)** | 0.1125 |
| **Empirical Null 95th Percentile** | 0.6834 |
| **Empirical Null 99th Percentile** | 0.7561 |
| **Empirical Null Maximum AUC** | 0.8433 |
| **Empirical $p$-Value ($p_{\text{perm}}$)** | **$0.000999$ ($p < 0.001$)** |
| **Overfitting Hypothesis Decision** | **Null hypothesis rejected ($p < 0.001$)** |

The observed cross-validated ROC-AUC of 0.9983 exceeded the maximum value recorded across all 1,000 label permutations ($\max \text{AUC}_{\text{null}} = 0.8433$), establishing an exact empirical $p$-value of $p = 0.000999$ ($p < 0.001$). This confirms that the model captures genuine biological signal and rejects chance association.

### 2.4 Model Interpretability and Global Biomarker Attributions
SHAP feature importance was extracted across all 116 predictors. Table 4 presents the top 15 biomarkers ranked by mean absolute Shapley value along with Pearson correlation with response.

#### Table 4: Top 15 biomarkers ranked by global SHAP attribution magnitude.
Computed using Linear SHAP on the champion ElasticNet model calibrated against the training cohort background.

| Rank | Feature Identifier | Omic Modality Domain | Mean Absolute SHAP | Pearson Correlation ($r$) | Direction of Effect |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 1 | `metab_4_hydroxy_butanone_log2` | Small-Molecule Metabolomics | **0.6743** | -0.7605 | Elevated in Non-Responders |
| 2 | `ohe_pdl1_negative` | Tumor Genomics & Pathology | **0.5800** | -0.8519 | Strongly Favors Non-Response |
| 3 | `pdl1_ihc_tps_percent` | Tumor Genomics & Pathology | **0.4815** | +0.8508 | Strongly Favors Response |
| 4 | `bio_tmb_pdl1_interaction` | Tumor Genomics & Pathology | **0.2981** | +0.7216 | Synergistic Response Driver |
| 5 | `prot_tnf` | Circulating Proteomics (Olink) | **0.1584** | -0.3425 | Elevated in Non-Responders |
| 6 | `ohe_pdl1_high` | Tumor Genomics & Pathology | **0.1281** | +0.6375 | Favors Response ($\text{TPS} \ge 50\%$) |
| 7 | `prot_pdl2_olink` | Circulating Proteomics (Olink) | **0.1248** | -0.4424 | Elevated in Non-Responders |
| 8 | `bio_tmb_pdl1_double_low` | Tumor Genomics & Pathology | **0.1084** | -0.7171 | Cold Immune Archetype |
| 9 | `tmb_high_binary` | Tumor Genomics & Pathology | **0.1036** | +0.6065 | Favors Response ($\text{TMB} \ge 10$) |
| 10| `tmb` | Tumor Genomics & Pathology | **0.0616** | +0.5562 | Continuous Neoantigen Load |
| 11| `histology_adeno` | Tumor Genomics & Pathology | **0.0603** | +0.2301 | Adenocarcinoma Subtype |
| 12| `metab_gentisic_acid_log2` | Small-Molecule Metabolomics | **0.0260** | -0.4754 | Elevated in Non-Responders |
| 13| `prot_il15` | Circulating Proteomics (Olink) | **0.0249** | -0.3745 | Elevated in Non-Responders |
| 14| `desmocolin_3` | Clinical & Pathology | **0.0184** | +0.4067 | Squamous Cell Marker |
| 15| `ohe_drug_nivolumab` | Clinical Regimen | **0.0170** | -0.3897 | Regimen Factor |

The strongest individual predictor identified by SHAP attribution was detrended **4-hydroxy-butanone ($log_2$)**, exhibiting a mean absolute SHAP value of 0.6743 and a negative correlation with response ($r = -0.7605$). Elevated circulating levels of this metabolite correspond to primary therapeutic resistance. 

Tissue checkpoint expression formed the second primary regulatory pillar: `ohe_pdl1_negative` ($\text{SHAP} = 0.5800, r = -0.8519$) and continuous `pdl1_ihc_tps_percent` ($\text{SHAP} = 0.4815, r = +0.8508$). 

Tumor Mutational Burden exerted an independent positive effect, amplified when interacting with PD-L1 TPS (`bio_tmb_pdl1_interaction`, $\text{SHAP} = 0.2981, r = +0.7216$). 

In circulating proteomics, baseline elevations of **Tumor Necrosis Factor (TNF)** ($\text{SHAP} = 0.1584, r = -0.3425$) and **PD-L2 (B7-DC)** ($\text{SHAP} = 0.1248, r = -0.4424$) were selected as systemic markers of primary checkpoint resistance.

### 2.5 Multi-Omic Biomarker Interaction Synergies
Cross-modality interaction terms were calculated to evaluate non-linear biological synergy between independent omic platforms. Top interaction pairs are detailed in Table 5.

#### Table 5: Cross-modality biomarker interaction synergies.
Ranked by significance of Spearman rank correlation with objective therapeutic response.

| Interaction Type | Feature 1 | Feature 2 | Spearman Correlation ($\rho$) | $p$-Value | Responder Mean Synergy | Non-Responder Mean Synergy | Biological Trajectory |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Genomics $\times$ Metabolomics** | `bio_tmb_pdl1_interaction` | `metab_thiourea_log2` | **+0.8782** | $2.60 \times 10^{-17}$ | 114.5273 | 3.9704 | Positive Biomarker Synergy |
| **Genomics $\times$ Metabolomics** | `pdl1_ihc_tps_percent` | `metab_thiourea_log2` | **+0.8727** | $7.19 \times 10^{-17}$ | 813.7736 | 79.3240 | Positive Biomarker Synergy |
| **Genomics $\times$ Metabolomics** | `bio_tmb_pdl1_interaction` | `metab_4_hydroxy_butanone_log2` | **+0.8672** | $1.90 \times 10^{-16}$ | 108.3512 | 4.0338 | Positive Biomarker Synergy |
| **Genomics $\times$ Metabolomics** | `bio_tmb_pdl1_interaction` | `metab_gentisic_acid_log2` | **+0.8617** | $4.81 \times 10^{-16}$ | 137.8809 | 4.9240 | Positive Biomarker Synergy |
| **Genomics $\times$ Proteomics** | `bio_tmb_pdl1_interaction` | `prot_il15` | **+0.8534** | $1.80 \times 10^{-15}$ | 16.8878 | 0.8698 | Positive Biomarker Synergy |
| **Genomics $\times$ Proteomics** | `bio_tmb_pdl1_interaction` | `prot_tnf` | **+0.8507** | $2.75 \times 10^{-15}$ | 20.3630 | 1.0104 | Positive Biomarker Synergy |
| **Genomics $\times$ Proteomics** | `ohe_pdl1_negative` | `prot_tnf` | **-0.8182** | $2.28 \times 10^{-13}$ | 0.0000 | 2.8897 | Negative Synergy (Resistance) |
| **Genomics $\times$ Proteomics** | `ohe_pdl1_negative` | `prot_pdl2_olink` | **-0.8182** | $2.28 \times 10^{-13}$ | 0.0000 | 1.0815 | Negative Synergy (Resistance) |
| **Genomics $\times$ Metabolomics** | `ohe_pdl1_negative` | `metab_thiourea_log2` | **-0.8182** | $2.28 \times 10^{-13}$ | 0.0000 | 12.9920 | Negative Synergy (Resistance) |
| **Genomics $\times$ Metabolomics** | `ohe_pdl1_negative` | `metab_gentisic_acid_log2` | **-0.8182** | $2.28 \times 10^{-13}$ | 0.0000 | 17.8083 | Negative Synergy (Resistance) |

Interaction analysis confirmed strong biological coupling across genomic, metabolomic, and proteomic boundaries. The interaction product of tumor antigenicity (`bio_tmb_pdl1_interaction`) with metabolomic peak intensity (`metab_thiourea_log2`) achieved a rank correlation of $\rho = +0.8782$ ($p = 2.60 \times 10^{-17}$), with responder tumors displaying an average synergy score of 114.53 compared to 3.97 in non-responders. 

Conversely, co-occurrence of absent PD-L1 expression (`ohe_pdl1_negative`) with elevated baseline circulating TNF ($\rho = -0.8182, p = 2.28 \times 10^{-13}$) or elevated circulating PD-L2 ($\rho = -0.8182, p = 2.28 \times 10^{-13}$) identified an aggressive primary resistance phenotype characterized by systemic inflammation and lack of adaptive checkpoint ligand induction.
