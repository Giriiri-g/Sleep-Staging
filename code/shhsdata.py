import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"C:\Users\Hari\Documents\Sleep-Staging\csv-docs\shhs1-dataset-0.21.0.csv"
df = pd.read_csv(csv_path)

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
df_clean = df[existing_columns]

patient_counts = {disease: (df_clean[disease] == 1).sum() for disease in existing_columns}

labels = [disease_id_to_name[disease] for disease in existing_columns]
values = [patient_counts[disease] for disease in existing_columns]

plt.figure(figsize=(13, 6))
bars = plt.bar(labels, values)
plt.xticks(rotation=90)
plt.xlabel('Disease Name')
plt.ylabel('Number of Patients')
plt.title('Patient Counts by Disease')
plt.tight_layout()

# Add numbers on top of each bar
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value),
             ha='center', va='bottom', fontsize=10)

plt.show()
