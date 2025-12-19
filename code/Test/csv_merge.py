import numpy as np
import pandas as pd

df1 = pd.read_csv('sleep_stages_output.csv')
df2 = pd.read_csv('csv-docs/mesa_selected.csv')

df2['mesaid'] = df2['path'].apply(lambda x: x.split('-')[-1].replace('.edf', ''))

df2['sleep_stages'] = np.where(df2['mesaid'].isin(df1['mesaid']), df1.set_index('mesaid').loc[df2['mesaid'], 'sleep_stages'], np.nan)
df2.to_csv('csv-docs/mesa_selected.csv', index=False)