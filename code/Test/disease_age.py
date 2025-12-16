import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- SHHS Plot Code ----------------
csv_path = r"csv-docs\shhs1-dataset-0.21.0.csv"
df = pd.read_csv(csv_path)

# Disease mapping
disease_id_to_name = {
    'angina15': 'Angina',
    'asthma15': 'Asthma',
    'ca15': 'Coronary Angioplasty',
    'cabg15': 'Coronary Artery Bypass Graft',
    'copd15': 'Chronic Obstructive Pulmonary Disease',
    'crbron15': 'Chronic Bronchitis',
    'emphys15': 'Emphysema',
    'hf15': 'Heart Failure',
    'mi15': 'Heart Attack (MI)',
    'sa15': 'Sleep Apnea',
    'prev_hx_mi': 'Previous History of MI',
    'prev_hx_stroke': 'Previous History of Stroke',
    'stroke15': 'Reported Stroke',
    'hi201a': 'Emphysema',
    'hi201b': 'Chronic Bronchitis',
    'hi201c': 'Chronic Obstructive Pulmonary Disease',
    'hi201d': 'Asthma',
    'hi201e': 'Current Asthma',
    'hi216': 'History of Restless Leg Syndrome',
    'htnderv_s1': 'Hypertension'
}

disease_ids = list(disease_id_to_name.keys())
existing_columns = [col for col in disease_ids if col in df.columns]
df_clean = df[existing_columns + ['age_s1']]  # include age column

# Filter into two age groups
under60 = df_clean[df_clean['age_s1'] < 60]
above60 = df_clean[df_clean['age_s1'] >= 60]

# Count patients for each disease in both groups
under60_counts = {disease: (under60[disease] == 1).sum() for disease in existing_columns}
above60_counts = {disease: (above60[disease] == 1).sum() for disease in existing_columns}

labels = [disease_id_to_name[disease] for disease in existing_columns]
under_values = [under60_counts[disease] for disease in existing_columns]
above_values = [above60_counts[disease] for disease in existing_columns]

# Bar plot with two bars per disease
x = range(len(labels))
bar_width = 0.4

plt.figure(figsize=(14, 6))
bars1 = plt.bar([i - bar_width/2 for i in x], under_values, width=bar_width, label='Under 60')
bars2 = plt.bar([i + bar_width/2 for i in x], above_values, width=bar_width, label='60 and Above')

plt.xticks(x, labels, rotation=90)
plt.xlabel('Disease Name')
plt.ylabel('Number of Patients')
plt.title('Patient Counts by Disease (Under 60 vs 60 and Above) - SHHS Dataset')
plt.legend()
plt.tight_layout()

# Add numbers on top of each bar
for bar in bars1 + bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(int(bar.get_height())),
             ha='center', va='bottom', fontsize=9)

plt.show()


# ---------------- Apples Dataset Plot ----------------
apples_csv_path = r"csv-docs\apples-dataset-0.1.0.csv"  # change path if needed
apples_df = pd.read_csv(apples_csv_path)

apples_disease_id_to_name = {
    'allergicrhinitismedhxhp': 'Allergic Rhinitis',
    'anemiamedhxhp': 'Anemia',
    'anxietymedhxhp': 'Anxiety',
    'asthmamedhxhp': 'Asthma',
    'bleedingdomedhxhp': 'Bleeding Disorder',
    'cancermedhxhp': 'Cancer',
    'ccbruxismhp': 'Bruxism',
    'ccnocturnalrefluxhp': 'Nocturnal Reflux',
    'chronicpainsyndmedhxhp': 'Chronic Pain Syndrome',
    'claustrophobiamedhxhp': 'Claustrophobia',
    'contactdermmedhxhp': 'Contact Dermatitis',
    'copdmedhxhp': 'Chronic Obstructive Pulmonary Disease',
    'depressionmedhxhp': 'Depression',
    'diabetesmedhxhp': 'Diabetes',
    'eczemamedhxhp': 'Eczema',
    'gerdmedhxhp': 'Gastroesophageal Reflux Disease',
    'hpcvangina': 'Angina',
    'hpcvcad': 'Coronary Artery Disease',
    'hpcvchf': 'Congenital Heart Failure',
    'hpcvdysrhyth': 'Dysrhythmia',
    'hpcvedema': 'Edema',
    'hpcvmi': 'Myocardial Infarction (MI)',
    'latexallergymedhxhp': 'Latex Allergy',
    'neuromuscdzmedhxhp': 'Neuromuscular Disease',
    'renalfailuremedhxhp': 'Renal Failure',
    'rhinoplastymedhxhp': 'Rhinoplasty',
    'strokemedhxhp': 'Stroke',
    'syncopemedhxhp': 'Syncope',
    'thyroiddzmedhxhp': 'Thyroid Disease'
}

# Extract only the disease columns that exist in the apples dataset
apples_disease_ids = list(apples_disease_id_to_name.keys())
apples_existing_cols = [col for col in apples_disease_ids if col in apples_df.columns]

# Filter only age >= 60
apples_above60 = apples_df[apples_df['age'] >= 60]

# Count patients with each disease above 60
apples_above60_counts = {disease: (apples_above60[disease] == 2).sum() for disease in apples_existing_cols}

# Plot only above 60 group
labels_apples = [apples_disease_id_to_name[disease] for disease in apples_existing_cols]
values_apples = [apples_above60_counts[disease] for disease in apples_existing_cols]

plt.figure(figsize=(14, 6))
bars = plt.bar(labels_apples, values_apples, color='orange', width=0.5)

plt.xticks(rotation=90)
plt.xlabel('Disease Name')
plt.ylabel('Number of Patients (Age ≥ 60)')
plt.title('Patient Counts by Disease (Age ≥ 60) - Apples Dataset')
plt.tight_layout()

# Add numbers on top of bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(int(bar.get_height())),
             ha='center', va='bottom', fontsize=9)

plt.show()

# ----------------WSC Dataset Plot (Under 60 vs 60 and Above) ----------------
wsc_csv_path = r"csv-docs\wsc-dataset-0.7.0.csv"  # change path if needed
wsc_df = pd.read_csv(wsc_csv_path)

wsc_disease_id_to_name = {
    "angina_ynd": "Angina",
    "arrhythmia_ynd": "Arrhythmia",
    "arthritis_ynd": "Arthritis",
    "asthma_ynd": "Asthma",
    "atheroscl_ynd": "Atherosclerosis",
    "congestivehf_ynd": "Congestive Heart Failure",
    "coronary_ynd": "Coronary Artery Disease",
    "diabetes_ynd": "Diabetes",
    "emphysema_ynd": "Emphysema/Obstructive Lung Disease",
    "heartattack_ynd": "Heart Attack",
    "hypertension_ynd": "Hypertension",
    "stroke_ynd": "Stroke",
    "thyroid_ynd": "Thyroid Disorder"
}

# Extract only relevant columns that exist
wsc_disease_ids = list(wsc_disease_id_to_name.keys())
wsc_existing_cols = [col for col in wsc_disease_ids if col in wsc_df.columns]

# Split into two age groups
wsc_under60 = wsc_df[wsc_df['age'] < 60]
wsc_above60 = wsc_df[wsc_df['age'] >= 60]

# Count number of patients with each disease in both groups
wsc_under60_counts = {d: (wsc_under60[d] == 'Y').sum() for d in wsc_existing_cols}
wsc_above60_counts = {d: (wsc_above60[d] == 'Y').sum() for d in wsc_existing_cols}

# Labels and values
labels_wsc = [wsc_disease_id_to_name[d] for d in wsc_existing_cols]
under_values_wsc = [wsc_under60_counts[d] for d in wsc_existing_cols]
above_values_wsc = [wsc_above60_counts[d] for d in wsc_existing_cols]

# Plot two bars per disease
x = range(len(labels_wsc))
bar_width = 0.4

plt.figure(figsize=(14, 6))
bars1 = plt.bar([i - bar_width/2 for i in x], under_values_wsc, width=bar_width, label='Under 60', color='skyblue')
bars2 = plt.bar([i + bar_width/2 for i in x], above_values_wsc, width=bar_width, label='60 and Above', color='teal')

plt.xticks(x, labels_wsc, rotation=90)
plt.xlabel('Disease Name')
plt.ylabel('Number of Patients')
plt.title('Patient Counts by Disease (Under 60 vs 60 and Above) - WSC Dataset')
plt.legend()
plt.tight_layout()

# Add text labels on bars
for bar in bars1 + bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(int(bar.get_height())),
             ha='center', va='bottom', fontsize=9)

plt.show()

#------------------------------------------------------------cfs---------------------------------------------------------------------------

# Load dataset
cfs_csv_path = r"csv-docs\cfs-visit5-dataset-0.7.0 (1).csv"
cfs_df = pd.read_csv(cfs_csv_path)

# Disease mapping
cfs_disease_id_to_name = {
    'cvd': 'cardiovascular disease',
    'cvdx': 'cardiovascular disease',
    'dementia': 'dementia',
    'diabetes2': 'diabetes',
    'diadiag': 'diabetes',
    'headiag': 'heart disease',
    'htfdiag': 'congestive heart failure',
    'irrdiag': 'irregular heartbeat',
    'kidfdiag': 'kidney failure',
    'parkdiag': 'Parkinson’s disease',
    'strodiag': 'stroke',
    'anemdiag': 'anemia',
    'angdiag': 'angina pectoris or chest pain from a heart condition',
    'bpdiag': 'high blood pressure',
    'brodiag': 'chronic bronchitis',
    'cancdiag': 'cancer',
    'cerebdisease': 'cerebrovascular disease'
}

# Separate under 60 and 60+ patients
age_cutoff = 60
age_counts_under = {}
age_counts_over = {}

for col, disease_name in cfs_disease_id_to_name.items():
    if col in cfs_df.columns:
        under_count = cfs_df[(cfs_df[col] == 1) & (cfs_df['age'] < age_cutoff)].shape[0]
        over_count = cfs_df[(cfs_df[col] == 1) & (cfs_df['age'] >= age_cutoff)].shape[0]
        age_counts_under[disease_name] = under_count
        age_counts_over[disease_name] = over_count

# Plotting
diseases = list(age_counts_under.keys())
under_60 = list(age_counts_under.values())
above_60 = list(age_counts_over.values())

x = np.arange(len(diseases))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))  # slightly bigger figure

bars1 = ax.bar(x - width/2, under_60, width, label='Under 60', color='#4A90E2', edgecolor='black')
bars2 = ax.bar(x + width/2, above_60, width, label='60 and Above', color='#50E3C2', edgecolor='black')

# Add value labels above bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Disease Name', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Patient Counts by Disease (Under 60 vs 60 and Above) - CFS Dataset',
             fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=10)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# ---------------- Load MROS Dataset ----------------
mros_csv_path = r"csv-docs\mros-visit1-dataset-0.6.0.csv"  # update with your path
mros_df = pd.read_csv(mros_csv_path)

# ---------------- Disease Dictionary ----------------
mros_elderly_disease_dict = {
    "cvaorane": "Aortic Aneurysm",
    "cvapcora": "Coronary Artery Disease",
    "cvcabg": "Coronary Artery Bypass Surgery",
    "cvcer": "Cerebrovascular Disease (Stroke)",
    "cvchd": "Coronary Heart Disease",
    "cvcp30m": "Prolonged Chest Pain (Angina/MI)",
    "cvtia": "Transient Ischemic Attack (Mini-Stroke)",
    "mhchf": "Congestive Heart Failure",
    "mhcobpd": "Chronic Obstructive Pulmonary Disease (COPD)",
    "mhdiab": "Diabetes Mellitus",
    "mhmi": "Myocardial Infarction (Heart Attack)",
    "mhrenal": "Chronic Kidney Disease / Renal Failure",
    "mhstrk": "Stroke / Brain Hemorrhage"
}

# ---------------- Compute counts ----------------
age_cutoff = 60
age_counts_under = {}
age_counts_over = {}

for col, disease_name in mros_elderly_disease_dict.items():
    if col in mros_df.columns:
        under_count = mros_df[(mros_df[col] == 1) & (mros_df['vsage1'] < age_cutoff)].shape[0]
        over_count = mros_df[(mros_df[col] == 1) & (mros_df['vsage1'] >= age_cutoff)].shape[0]
        age_counts_under[disease_name] = under_count
        age_counts_over[disease_name] = over_count

# ---------------- Plotting ----------------
diseases = list(age_counts_under.keys())
under_60 = list(age_counts_under.values())
above_60 = list(age_counts_over.values())

x = np.arange(len(diseases))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))

bars1 = ax.bar(x - width/2, under_60, width, label='Under 60', color='#FFA500', edgecolor='black')
bars2 = ax.bar(x + width/2, above_60, width, label='60 and Above', color='#2E8B57', edgecolor='black')

# Add value labels above bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Disease Name', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('MROS Patient Counts by Disease (Under 60 vs 60 and Above)',
             fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=10)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


mros_csv_path = r"csv-docs\mros-visit1-dataset-0.6.0.csv"  # update with your path
mros_df = pd.read_csv(mros_csv_path)

# ---------------- Disease Dictionary ----------------
mros_elderly_disease_dict = {
    "cvaorane": "Aortic Aneurysm",
    "cvapcora": "Coronary Artery Disease",
    "cvcabg": "Coronary Artery Bypass Surgery",
    "cvcer": "Cerebrovascular Disease (Stroke)",
    "cvchd": "Coronary Heart Disease",
    "cvcp30m": "Prolonged Chest Pain (Angina/MI)",
    "cvtia": "Transient Ischemic Attack (Mini-Stroke)",
    "mhchf": "Congestive Heart Failure",
    "mhcobpd": "Chronic Obstructive Pulmonary Disease (COPD)",
    "mhdiab": "Diabetes Mellitus",
    "mhmi": "Myocardial Infarction (Heart Attack)",
    "mhrenal": "Chronic Kidney Disease / Renal Failure",
    "mhstrk": "Stroke / Brain Hemorrhage"
}

# ---------------- Compute counts ----------------
age_cutoff = 60
age_counts_under = {}
age_counts_over = {}

for col, disease_name in mros_elderly_disease_dict.items():
    if col in mros_df.columns:
        under_count = mros_df[(mros_df[col] == 1) & (mros_df['vsage1'] < age_cutoff)].shape[0]
        over_count = mros_df[(mros_df[col] == 1) & (mros_df['vsage1'] >= age_cutoff)].shape[0]
        age_counts_under[disease_name] = under_count
        age_counts_over[disease_name] = over_count

# ---------------- Plotting ----------------
diseases = list(age_counts_under.keys())
under_60 = list(age_counts_under.values())
above_60 = list(age_counts_over.values())

x = np.arange(len(diseases))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))

bars1 = ax.bar(x - width/2, under_60, width, label='Under 60', color='#FFA500', edgecolor='black')
bars2 = ax.bar(x + width/2, above_60, width, label='60 and Above', color='#2E8B57', edgecolor='black')

# Add value labels above bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{int(height)}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Disease Name', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('MROS Patient Counts by Disease (Under 60 vs 60 and Above)',
             fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=10)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
