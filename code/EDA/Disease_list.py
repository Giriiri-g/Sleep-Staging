## List the diseases available in each dataset
import pandas as pd



cfs = pd.read_csv('dictionary/cfs-data-dictionary-0.7.0-variables.csv')
apples = pd.read_csv('dictionary/apples-data-dictionary-0.1.0-variables.csv')
mesa = pd.read_csv('dictionary/mesa-data-dictionary-0.8.0-variables.csv')
shhs = pd.read_csv('dictionary/shhs-data-dictionary-0.21.0-variables.csv')
mros = pd.read_csv('dictionary/mros-data-dictionary-0.6.0-variables.csv')
wsc = pd.read_csv('dictionary/wsc-data-dictionary-0.7.0-variables (1).csv')

def get_diseases(df, name):
    diseases_dict = {}

    for _, row in df.iterrows():
        if name =='cfs':
            if 'diag' in row.iloc[1].lower():
                diseases_dict[row.iloc[1]] = row.iloc[2]
        elif name == 'apples':
            if 'ongoing' in row.iloc[2].lower():
                diseases_dict[row.iloc[1]] = row.iloc[2]
        elif name == 'mesa':
            if 'diagnosis' in row.iloc[2].lower() or 'urgent' in row.iloc[2].lower():
                diseases_dict[row.iloc[1]] = row.iloc[2]
        elif name == 'shhs':   
            if not pd.isnull(row.iloc[3]) and 'Doctor of Medicine' in row.iloc[3]:
                diseases_dict[row.iloc[1]] = row.iloc[2]
        elif name == 'mros':
            if 'are you currently' in row.iloc[2].lower():
                diseases_dict[row.iloc[1]] = row.iloc[2]
        elif name == 'wsc':
            if 'ynd' in row.iloc[1].lower():
                diseases_dict[row.iloc[1]] = row.iloc[2]
    return diseases_dict

cfs_disease = get_diseases(cfs, 'cfs')
apples_disease = get_diseases(apples, 'apples')
mesa_disease = get_diseases(mesa, 'mesa')
shhs_disease = get_diseases(shhs, 'shhs')
mros_disease = get_diseases(mros, 'mros')
wsc_disease = get_diseases(wsc, 'wsc')

# print("CFS Diseases:", cfs_disease)
# print("Apples Diseases:", apples_disease)
# print("MESA Diseases:", mesa_disease)
# print("SHHS Diseases:", shhs_disease)
# print("MROS Diseases:", mros_disease)
# print("WSC Diseases:", wsc_disease)

# Export dictionaries to a single Excel sheet
with pd.ExcelWriter('diseases_summary.xlsx') as writer:
    combined_data = []
    combined_data.extend([("CFS", k, v) for k, v in cfs_disease.items()])
    combined_data.extend([("Apples", k, v) for k, v in apples_disease.items()])
    combined_data.extend([("MESA", k, v) for k, v in mesa_disease.items()])
    combined_data.extend([("SHHS", k, v) for k, v in shhs_disease.items()])
    combined_data.extend([("MROS", k, v) for k, v in mros_disease.items()])
    combined_data.extend([("WSC", k, v) for k, v in wsc_disease.items()])

    pd.DataFrame(combined_data, columns=['Dataset', 'ID', 'Name']).to_excel(writer, index=False, sheet_name='Diseases Summary')