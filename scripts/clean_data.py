import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats

def clean_ici_dataset(project_root=r'c:\Users\SAM\Documents\GitHub\SGRH-Lung Cancer', randomize_rows=True, seed=42):
    raw_file = os.path.join(project_root, 'data', 'raw', 'ICI.xlsx')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load raw dataset in read-only mode
    print(f"Loading raw dataset from: {raw_file}")
    df_raw = pd.read_excel(raw_file)
    n_rows, n_cols = df_raw.shape
    print(f"Loaded raw data: {n_rows} rows x {n_cols} columns")
    
    # Initialize cleaned DataFrame
    df_clean = pd.DataFrame()
    data_dict = []
    
    # --- A. SAMPLE IDENTIFIER & CLINICAL RESPONSE ---
    sample_col = df_raw.columns[0]
    df_clean['sample_id'] = df_raw[sample_col].astype(int)
    data_dict.append({
        'clean_name': 'sample_id',
        'raw_name': sample_col,
        'dtype': 'int',
        'category': 'Identifier',
        'description': 'Unique patient sequential identifier (1..64). WARNING: Confounded with outcome sorting.'
    })
    
    label_col = df_raw.columns[1]
    df_clean['response_label'] = df_raw[label_col].astype(str).str.strip()
    df_clean['response_binary'] = (df_clean['response_label'] == 'R').astype(int)
    data_dict.append({
        'clean_name': 'response_label',
        'raw_name': label_col,
        'dtype': 'string',
        'category': 'Target',
        'description': 'Clinical response to ICI: R (Responder, n=34), NR (Non-responder, n=30).'
    })
    data_dict.append({
        'clean_name': 'response_binary',
        'raw_name': label_col,
        'dtype': 'binary (0/1)',
        'category': 'Target',
        'description': 'Binary target variable: 1 = Responder (R), 0 = Non-responder (NR).'
    })
    
    # --- B. DEMOGRAPHICS & ANTHROPOMETRICS ---
    df_clean['age'] = df_raw['age'].astype(int)
    data_dict.append({'clean_name': 'age', 'raw_name': 'age', 'dtype': 'int', 'category': 'Demographic', 'description': 'Patient age in years.'})
    
    # Fix whitespace in sex
    cleaned_sex = df_raw['sex'].astype(str).str.strip().str.upper()
    df_clean['sex'] = cleaned_sex
    df_clean['sex_male'] = (cleaned_sex == 'M').astype(int)
    data_dict.append({'clean_name': 'sex', 'raw_name': 'sex', 'dtype': 'category (M/F)', 'category': 'Demographic', 'description': 'Biological sex, cleaned of trailing whitespace (Sample 10 was "F ").'})
    data_dict.append({'clean_name': 'sex_male', 'raw_name': 'sex', 'dtype': 'binary (0/1)', 'category': 'Demographic', 'description': 'Binary indicator: 1 = Male (n=42), 0 = Female (n=22).'})
    
    df_clean['height_cm'] = df_raw['height'].astype(float)
    df_clean['weight_kg'] = df_raw['weight'].astype(float)
    df_clean['bmi_recorded'] = df_raw['BMI'].astype(int)
    
    # Recalculate true clinical BMI
    calc_bmi = df_clean['weight_kg'] / ((df_clean['height_cm'] / 100.0) ** 2)
    df_clean['bmi_calculated'] = np.round(calc_bmi, 2)
    df_clean['bmi_discrepancy'] = np.round(df_clean['bmi_recorded'] - calc_bmi, 2)
    
    # WHO BMI Classification
    conditions = [
        (df_clean['bmi_calculated'] < 18.5),
        (df_clean['bmi_calculated'] >= 18.5) & (df_clean['bmi_calculated'] < 25.0),
        (df_clean['bmi_calculated'] >= 25.0) & (df_clean['bmi_calculated'] < 30.0),
        (df_clean['bmi_calculated'] >= 30.0)
    ]
    choices = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_clean['bmi_who_category'] = np.select(conditions, choices, default='Unknown')
    
    data_dict.extend([
        {'clean_name': 'height_cm', 'raw_name': 'height', 'dtype': 'float', 'category': 'Anthropometric', 'description': 'Height in centimeters.'},
        {'clean_name': 'weight_kg', 'raw_name': 'weight', 'dtype': 'float', 'category': 'Anthropometric', 'description': 'Weight in kilograms.'},
        {'clean_name': 'bmi_recorded', 'raw_name': 'BMI', 'dtype': 'int', 'category': 'Anthropometric', 'description': 'Original recorded integer BMI.'},
        {'clean_name': 'bmi_calculated', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Anthropometric', 'description': 'Mathematically verified clinical BMI: weight / (height/100)^2.'},
        {'clean_name': 'bmi_discrepancy', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Anthropometric', 'description': 'Difference: bmi_recorded - bmi_calculated.'},
        {'clean_name': 'bmi_who_category', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Anthropometric', 'description': 'WHO BMI category based on calculated BMI.'}
    ])
    
    # --- C. DIET & HABITS ---
    # Fix backtick typo in non-veg
    raw_nonveg = df_raw['non-veg.'].astype(str).str.strip().str.replace('`', '')
    clean_nonveg = raw_nonveg.astype(int)
    clean_veg = df_raw['Veg.'].astype(int)
    
    df_clean['diet_veg_flag'] = clean_veg
    df_clean['diet_nonveg_flag'] = clean_nonveg
    
    # Consolidated Diet Category
    # If non_veg == 1, patient consumes non-vegetarian food regardless of veg flag
    df_clean['diet_category'] = np.where(clean_nonveg == 1, 'Non-Vegetarian', 'Vegetarian')
    df_clean['diet_is_nonveg'] = (df_clean['diet_category'] == 'Non-Vegetarian').astype(int)
    
    data_dict.extend([
        {'clean_name': 'diet_veg_flag', 'raw_name': 'Veg.', 'dtype': 'binary (0/1)', 'category': 'Lifestyle', 'description': 'Original recorded vegetarian flag.'},
        {'clean_name': 'diet_nonveg_flag', 'raw_name': 'non-veg.', 'dtype': 'binary (0/1)', 'category': 'Lifestyle', 'description': 'Cleaned non-vegetarian flag (fixed Sample 40 typo "1`").'},
        {'clean_name': 'diet_category', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Lifestyle', 'description': 'Consolidated diet: Vegetarian (n=28) vs Non-Vegetarian (n=36).'},
        {'clean_name': 'diet_is_nonveg', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'Lifestyle', 'description': 'Binary indicator for non-vegetarian diet.'}
    ])
    
    # Habits & Comorbidities
    comorbid_map = {
        'Diabetes mellitus (DM)': 'comorbidity_diabetes',
        'Hypertension ': 'comorbidity_hypertension',
        'Any lung or heart diseases': 'comorbidity_lung_heart_disease',
        'SMOKING ': 'habit_smoking',
        'ALCOHOL': 'habit_alcohol',
        'TABACCO CHEWING': 'habit_tobacco_chewing',
        'WEIGHT LOSS ': 'symptom_weight_loss',
        'PLEURAL EFFUSION ': 'sign_pleural_effusion',
        'PRE-TREATED RADIOTHERAPY ': 'prior_radiotherapy',
        'PRE-TREATED CHEMOTHERAPY ': 'prior_chemotherapy',
        'POST -SURGERY ': 'prior_surgery'
    }
    for raw_k, clean_k in comorbid_map.items():
        df_clean[clean_k] = df_raw[raw_k].astype(int)
        data_dict.append({'clean_name': clean_k, 'raw_name': raw_k, 'dtype': 'binary (0/1)', 'category': 'Clinical/Comorbidity', 'description': f'Binary status for {clean_k}.'})
    
    # --- D. HISTOLOGY & STAGING ---
    df_clean['histology_adeno'] = df_raw['ADENOCARCINOMA '].astype(int)
    df_clean['histology_squamous'] = df_raw['SQUAMOUS CELL CARCINOMA '].astype(int)
    df_clean['histology_large_cell'] = df_raw['LARGE CELL CARCINOMA '].astype(int)
    df_clean['cancer_type_raw'] = df_raw['CANCER/TYPE '].astype(int)
    
    # Build consolidated histology classification
    h_sum = df_clean['histology_adeno'] + df_clean['histology_squamous'] + df_clean['histology_large_cell']
    df_clean['histology_flags_count'] = h_sum
    
    def resolve_histology(row):
        adeno = row['histology_adeno']
        sq = row['histology_squamous']
        lc = row['histology_large_cell']
        if adeno == 1 and sq == 0 and lc == 0:
            return 'Adenocarcinoma'
        elif sq == 1 and adeno == 0 and lc == 0:
            return 'Squamous Cell Carcinoma'
        elif lc == 1 and adeno == 0 and sq == 0:
            return 'Large Cell Carcinoma'
        elif (adeno + sq + lc) > 1:
            return 'Adenosquamous / Mixed'
        else:
            return 'NSCLC - Unspecified'
            
    df_clean['histology_consensus'] = df_clean.apply(resolve_histology, axis=1)
    
    # Staging & Metastasis
    df_clean['disease_stage'] = df_raw['DISEASE STAGE '].astype(str).str.strip().str.upper()
    df_clean['is_stage_iv'] = (df_clean['disease_stage'] == 'IV').astype(int)
    df_clean['metastasis'] = df_raw['Metastasis'].astype(int)
    # Staging contradiction flag (Pt 22: Stage III but Metastasis == 1)
    df_clean['stage_metastasis_inconsistency'] = ((df_clean['disease_stage'] == 'III') & (df_clean['metastasis'] == 1)).astype(int)
    
    data_dict.extend([
        {'clean_name': 'histology_adeno', 'raw_name': 'ADENOCARCINOMA ', 'dtype': 'binary (0/1)', 'category': 'Histology', 'description': 'Adenocarcinoma binary flag.'},
        {'clean_name': 'histology_squamous', 'raw_name': 'SQUAMOUS CELL CARCINOMA ', 'dtype': 'binary (0/1)', 'category': 'Histology', 'description': 'Squamous cell carcinoma binary flag.'},
        {'clean_name': 'histology_large_cell', 'raw_name': 'LARGE CELL CARCINOMA ', 'dtype': 'binary (0/1)', 'category': 'Histology', 'description': 'Large cell carcinoma binary flag.'},
        {'clean_name': 'cancer_type_raw', 'raw_name': 'CANCER/TYPE ', 'dtype': 'binary (0/1)', 'category': 'Histology', 'description': 'Original binary column CANCER/TYPE (0=39, 1=25).'},
        {'clean_name': 'histology_flags_count', 'raw_name': 'Derived', 'dtype': 'int', 'category': 'Histology', 'description': 'Number of positive histology flags (0, 1, or 2).'},
        {'clean_name': 'histology_consensus', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Histology', 'description': 'Consolidated single-label histology subtype.'},
        {'clean_name': 'disease_stage', 'raw_name': 'DISEASE STAGE ', 'dtype': 'category (III/IV)', 'category': 'Staging', 'description': 'AJCC clinical disease stage.'},
        {'clean_name': 'is_stage_iv', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'Staging', 'description': 'Binary indicator for Stage IV disease.'},
        {'clean_name': 'metastasis', 'raw_name': 'Metastasis', 'dtype': 'binary (0/1)', 'category': 'Staging', 'description': 'Distant metastasis flag.'},
        {'clean_name': 'stage_metastasis_inconsistency', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'QualityFlag', 'description': 'Flag for Patient 22 (Stage III with Metastasis=1).'}
    ])
    
    # --- E. BIOMARKERS & PHARMACOTHERAPY ---
    df_clean['tmb'] = df_raw['TMB (>10 is considered for therapy'].astype(float)
    df_clean['tmb_high_binary'] = (df_clean['tmb'] >= 10.0).astype(int)
    df_clean['msi_status_raw'] = df_raw['MSI (MSI-L=0; MSS= 1)'].astype(int)
    df_clean['msi_category'] = np.where(df_clean['msi_status_raw'] == 1, 'MSS', 'MSI-L')
    df_clean['pdl1_ihc_tps_percent'] = df_raw['PDL-1 EXPRESSION '].astype(int)
    df_clean['pdl1_ihc_category'] = pd.cut(
        df_clean['pdl1_ihc_tps_percent'], 
        bins=[-1, 0, 49, 100], 
        labels=['Negative (0%)', 'Low/Intermediate (1-49%)', 'High (>=50%)']
    ).astype(str)
    df_clean['desmocolin_3'] = df_raw['DESM COL 3 (0=ABSENT; 1= PRESENT)'].astype(int)
    df_clean['tp53_mutation'] = df_raw['TP53p Gene Mutation (0=ABSENT; 1= PRESENT)'].astype(int)
    
    # ICI Drug Allocation
    df_clean['drug_nivolumab'] = df_raw['NIVOLUMAB '].astype(int)
    df_clean['drug_pembrolizumab'] = df_raw['PEMBROLIZUMAB '].astype(int)
    df_clean['ici_drug_name'] = np.where(df_clean['drug_nivolumab'] == 1, 'Nivolumab', 'Pembrolizumab')
    
    data_dict.extend([
        {'clean_name': 'tmb', 'raw_name': 'TMB (>10 is considered for therapy', 'dtype': 'float', 'category': 'Genomics', 'description': 'Tumor Mutational Burden (mut/Mb).'},
        {'clean_name': 'tmb_high_binary', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'Genomics', 'description': 'Standard clinical cutoff: 1 if TMB >= 10 mut/Mb, 0 otherwise.'},
        {'clean_name': 'msi_status_raw', 'raw_name': 'MSI (MSI-L=0; MSS= 1)', 'dtype': 'binary (0/1)', 'category': 'Genomics', 'description': 'MSI status: 0=MSI-L, 1=MSS.'},
        {'clean_name': 'msi_category', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Genomics', 'description': 'Categorical MSI status (MSS vs MSI-L).'},
        {'clean_name': 'pdl1_ihc_tps_percent', 'raw_name': 'PDL-1 EXPRESSION ', 'dtype': 'int', 'category': 'Pathology', 'description': 'IHC Tumor Proportion Score (TPS %: discrete 0, 30, 60). Distinct from Olink PD-L1 protein.'},
        {'clean_name': 'pdl1_ihc_category', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Pathology', 'description': 'Standard clinical TPS grouping (Negative, Intermediate, High).'},
        {'clean_name': 'desmocolin_3', 'raw_name': 'DESM COL 3 (0=ABSENT; 1= PRESENT)', 'dtype': 'binary (0/1)', 'category': 'Pathology', 'description': 'Desmocollin-3 expression status (0=Absent, 1=Present).'},
        {'clean_name': 'tp53_mutation', 'raw_name': 'TP53p Gene Mutation (0=ABSENT; 1= PRESENT)', 'dtype': 'binary (0/1)', 'category': 'Genomics', 'description': 'TP53 gene mutation status (0=Absent, 1=Present).'},
        {'clean_name': 'drug_nivolumab', 'raw_name': 'NIVOLUMAB ', 'dtype': 'binary (0/1)', 'category': 'Pharmacotherapy', 'description': 'Administered Nivolumab flag.'},
        {'clean_name': 'drug_pembrolizumab', 'raw_name': 'PEMBROLIZUMAB ', 'dtype': 'binary (0/1)', 'category': 'Pharmacotherapy', 'description': 'Administered Pembrolizumab flag.'},
        {'clean_name': 'ici_drug_name', 'raw_name': 'Derived', 'dtype': 'category', 'category': 'Pharmacotherapy', 'description': 'Administered ICI drug: Nivolumab (n=38) vs Pembrolizumab (n=26).'}
    ])
    
    # --- F. SURVIVAL ENDPOINTS & CLINICAL LOGIC ---
    df_clean['pfs_days_raw'] = df_raw['PFS (DAYS)'].astype(int)
    df_clean['os_days_raw'] = df_raw['OS (DAYS)'].astype(int)
    
    # Flag the 4 patients violating PFS <= OS
    pfs_os_inversion = (df_clean['pfs_days_raw'] > df_clean['os_days_raw']).astype(int)
    df_clean['pfs_os_inversion_flag'] = pfs_os_inversion
    
    # Harmonized PFS capped at OS (for survival models requiring PFS <= OS)
    df_clean['pfs_days_harmonized'] = np.minimum(df_clean['pfs_days_raw'], df_clean['os_days_raw'])
    
    # Recurrence & Vital Status
    df_clean['recurrence_event'] = df_raw['RECC. YES(1)/NO(0)'].astype(int)
    df_clean['vital_status_raw'] = df_raw['DEATH (0)/ALIVE(1)'].astype(int)
    
    # Standard survival coding: Event = 1 (Death), Censored = 0 (Alive)
    df_clean['death_event'] = (df_clean['vital_status_raw'] == 0).astype(int)
    
    data_dict.extend([
        {'clean_name': 'pfs_days_raw', 'raw_name': 'PFS (DAYS)', 'dtype': 'int', 'category': 'Survival', 'description': 'Raw Progression-Free Survival in days.'},
        {'clean_name': 'os_days_raw', 'raw_name': 'OS (DAYS)', 'dtype': 'int', 'category': 'Survival', 'description': 'Raw Overall Survival in days.'},
        {'clean_name': 'pfs_os_inversion_flag', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'QualityFlag', 'description': 'Indicator for 4 patients where raw PFS > OS (Pts 1, 21, 28, 53).'},
        {'clean_name': 'pfs_days_harmonized', 'raw_name': 'Derived', 'dtype': 'int', 'category': 'Survival', 'description': 'Harmonized PFS capped at OS (min(PFS, OS)) to satisfy oncological constraint.'},
        {'clean_name': 'recurrence_event', 'raw_name': 'RECC. YES(1)/NO(0)', 'dtype': 'binary (0/1)', 'category': 'Survival', 'description': 'Tumor recurrence event: 1=Yes (n=53), 0=No (n=11).'},
        {'clean_name': 'vital_status_raw', 'raw_name': 'DEATH (0)/ALIVE(1)', 'dtype': 'binary (0/1)', 'category': 'Survival', 'description': 'Original vital status coding: 0=Death, 1=Alive.'},
        {'clean_name': 'death_event', 'raw_name': 'Derived', 'dtype': 'binary (0/1)', 'category': 'Survival', 'description': 'Standard survival analysis event coding: 1 = Death (n=51), 0 = Censored/Alive (n=13).'}
    ])
    
    # --- G. HEMATOLOGY & INFLAMMATORY RATIOS ---
    hem_raw_map = {
        'TOTAL LEUCOCYTES COUNT ': 'tlc_base',
        'NEUTROPHILS ': 'neutrophils_base',
        'LYPHOCYTES PERCENTAGE ': 'lymphocytes_pct_base',
        'MONOCYTES ': 'monocytes_base',
        'EOSINOPHILS ': 'eosinophils_base',
        'TOTAL LEUCOCYTES COUNT .1': 'tlc_post',
        'NEUTROPHILS .1': 'neutrophils_post',
        'LYPHOCYTES PERCENTAGE .1': 'lymphocytes_pct_post',
        'MONOCYTES .1': 'monocytes_post',
        'EOSINOPHILS .1': 'eosinophils_post'
    }
    for raw_k, clean_k in hem_raw_map.items():
        df_clean[clean_k] = np.round(df_raw[raw_k].astype(float), 2)
        data_dict.append({'clean_name': clean_k, 'raw_name': raw_k, 'dtype': 'float', 'category': 'Hematology', 'description': f'Complete blood count parameter: {clean_k}.'})
        
    # Verified NLR calculations
    df_clean['nlr_base_recorded'] = np.round(df_raw['BASELINE RATIO'].astype(float), 2)
    df_clean['nlr_post_recorded'] = np.round(df_raw['AFTER THERAPY RATIO '].astype(float), 2)
    
    calc_nlr_b = df_clean['neutrophils_base'] / df_clean['lymphocytes_pct_base']
    calc_nlr_p = df_clean['neutrophils_post'] / df_clean['lymphocytes_pct_post']
    df_clean['nlr_base_calculated'] = np.round(calc_nlr_b, 4)
    df_clean['nlr_post_calculated'] = np.round(calc_nlr_p, 4)
    df_clean['nlr_delta'] = np.round(df_clean['nlr_post_calculated'] - df_clean['nlr_base_calculated'], 4)
    df_clean['nlr_ratio_change'] = np.round(df_clean['nlr_post_calculated'] / df_clean['nlr_base_calculated'], 4)
    
    # Differential sum check
    df_clean['diff_sum_base'] = np.round(df_clean['neutrophils_base'] + df_clean['lymphocytes_pct_base'] + df_clean['monocytes_base'] + df_clean['eosinophils_base'], 2)
    df_clean['diff_sum_post'] = np.round(df_clean['neutrophils_post'] + df_clean['lymphocytes_pct_post'] + df_clean['monocytes_post'] + df_clean['eosinophils_post'], 2)
    
    data_dict.extend([
        {'clean_name': 'nlr_base_recorded', 'raw_name': 'BASELINE RATIO', 'dtype': 'float', 'category': 'Hematology', 'description': 'Recorded baseline Neutrophil-to-Lymphocyte Ratio.'},
        {'clean_name': 'nlr_post_recorded', 'raw_name': 'AFTER THERAPY RATIO ', 'dtype': 'float', 'category': 'Hematology', 'description': 'Recorded post-therapy Neutrophil-to-Lymphocyte Ratio.'},
        {'clean_name': 'nlr_base_calculated', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Mathematically verified baseline NLR (neutrophils / lymphocytes).'},
        {'clean_name': 'nlr_post_calculated', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Mathematically verified post-therapy NLR.'},
        {'clean_name': 'nlr_delta', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Kinetics: nlr_post - nlr_base.'},
        {'clean_name': 'nlr_ratio_change', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Kinetics: nlr_post / nlr_base.'},
        {'clean_name': 'diff_sum_base', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Leukocyte differential percentage sum at baseline (N+L+M+E).'},
        {'clean_name': 'diff_sum_post', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Hematology', 'description': 'Leukocyte differential percentage sum post-therapy (N+L+M+E).'}
    ])
    
    # --- H. SMALL-MOLECULE METABOLOMICS (5-PLEX) ---
    metab_raw_cols = ['Thio-Urea', 'Gentisic acid', 'Sphingosine 1-phosphate', 'glutathione', '4-Hydroxy-butanone']
    metab_clean_names = ['thiourea', 'gentisic_acid', 'sphingosine_1_phosphate', 'glutathione', '4_hydroxy_butanone']
    
    for raw_m, clean_m in zip(metab_raw_cols, metab_clean_names):
        vals = df_raw[raw_m].astype(float)
        df_clean[f'metab_{clean_m}_raw'] = vals
        df_clean[f'metab_{clean_m}_log2'] = np.round(np.log2(vals), 4)
        
        # Detrending against sample run order using linear/quadratic polynomial regression
        # This provides a normalized version that removes the instrument signal drift
        sample_idx = df_clean['sample_id'].values
        poly_fit = np.polyfit(sample_idx, vals, deg=2)
        trend = np.polyval(poly_fit, sample_idx)
        residuals = vals - trend + np.mean(vals)
        df_clean[f'metab_{clean_m}_detrended'] = np.round(residuals, 2)
        
        data_dict.append({'clean_name': f'metab_{clean_m}_raw', 'raw_name': raw_m, 'dtype': 'float', 'category': 'Metabolomics', 'description': f'Raw mass-spec peak area for {raw_m}.'})
        data_dict.append({'clean_name': f'metab_{clean_m}_log2', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Metabolomics', 'description': f'log2-transformed peak area for {raw_m}.'})
        data_dict.append({'clean_name': f'metab_{clean_m}_detrended', 'raw_name': 'Derived', 'dtype': 'float', 'category': 'Metabolomics', 'description': f'Run-order detrended peak area for {raw_m} to correct instrument drift.'})
        
    # --- I. MULTIPLEX IMMUNO-ONCOLOGY PROTEOMICS (92-PLEX) ---
    prot_raw_cols = list(df_raw.columns[54:146])
    
    def sanitize_prot_name(col_name):
        c = col_name.strip()
        # Disambiguate Olink PD-L1 from IHC PD-L1
        if c == 'PD-L1':
            return 'prot_pdl1_olink'
        if c == 'PD-L2':
            return 'prot_pdl2_olink'
        # Replace non-alphanumeric with underscore
        clean = re.sub(r'[^a-zA-Z0-9]+', '_', c).strip('_').lower()
        return f'prot_{clean}'
        
    for raw_p in prot_raw_cols:
        clean_p = sanitize_prot_name(raw_p)
        vals = np.round(df_raw[raw_p].astype(float), 4)
        df_clean[clean_p] = vals
        
        data_dict.append({
            'clean_name': clean_p,
            'raw_name': raw_p,
            'dtype': 'float',
            'category': 'Proteomics',
            'description': f'Olink Normalized Protein Expression (NPX, log2 scale) for {raw_p}.'
        })
        
    # --- J. VALIDATION CHECKS ---
    print(f"\nFinal Cleaned Dimensions: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    print(f"Missing values across entire cleaned dataset: {df_clean.isna().sum().sum()}")
    assert df_clean.isna().sum().sum() == 0, "Cleaned dataset contains unexpected NaNs!"
    assert df_clean['sample_id'].nunique() == 64, "Sample IDs are not unique!"
    assert set(df_clean['sex'].unique()) == {'M', 'F'}, "Sex contains invalid categories!"
    assert set(df_clean['response_binary'].unique()) == {0, 1}, "Response binary is invalid!"
    
    # Optional row randomization to eliminate ascending sample/outcome order bias
    if randomize_rows:
        print(f"Randomizing row order (seed={seed}) to eliminate sequential outcome bias...")
        df_clean = df_clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"First 5 randomized sample_ids: {df_clean['sample_id'].iloc[:5].tolist()}")
        print(f"First 5 randomized response_labels: {df_clean['response_label'].iloc[:5].tolist()}")

    # --- K. EXPORT DATASETS ---
    xlsx_processed = os.path.join(processed_dir, 'ICI_cleaned.xlsx')
    csv_processed = os.path.join(processed_dir, 'ICI_cleaned.csv')
    xlsx_data_root = os.path.join(project_root, 'data', 'ICI_cleaned.xlsx')
    xlsx_raw_dir = os.path.join(raw_dir, 'ICI_cleaned.xlsx')
    dict_file = os.path.join(processed_dir, 'data_dictionary.json')
    
    # Save Excel and CSV
    df_clean.to_excel(xlsx_processed, index=False)
    df_clean.to_csv(csv_processed, index=False)
    df_clean.to_excel(xlsx_data_root, index=False)
    df_clean.to_excel(xlsx_raw_dir, index=False)
    
    # Save Data Dictionary
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2)
        
    print(f"Cleaned Excel saved to: {xlsx_processed}")
    print(f"Cleaned CSV saved to: {csv_processed}")
    print(f"Root Excel copy saved to: {xlsx_data_root}")
    print(f"Raw directory copy saved to: {xlsx_raw_dir}")
    print(f"Data Dictionary saved to: {dict_file}")
    
    return df_clean

if __name__ == '__main__':
    clean_ici_dataset()
