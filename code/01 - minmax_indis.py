import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/SDR_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]


df_meta = pd.read_excel(home+"data/raw/indicators/SDR 2021 - Database.xlsx", sheet_name='Codebook')


min_vals = []
max_vals = []
for index, row in df_meta.iterrows():
    dft = df[df.seriesCode==row.IndCode]
    min_val, max_val = np.nan, np.nan 
    if len(dft) > 0:
        min_val = np.nanmin(dft[colYears].values)
        max_val = np.nanmax(dft[colYears].values)
    min_vals.append(min_val)
    max_vals.append(max_val)


df_meta['min_vals'] = min_vals
df_meta['max_vals'] = max_vals


dft = df[df.seriesCode=='sdg8_gdpgrowth']
min_val, max_val = np.nan, np.nan 
min_val = np.nanmin(dft[colYears].values)
max_val = np.nanmax(dft[colYears].values)
min_vals.append(min_val)
max_vals.append(max_val)

new_row = dict(zip(df_meta.columns, [np.nan for i in range(len(df_meta.columns))]))
new_row['IndCode'] = 'sdg8_gdpgrowth'
new_row['SDG'] = 8
new_row['min_vals'] = min_val
new_row['max_vals'] = max_val
df_meta = pd.concat([df_meta, pd.Series(new_row)], ignore_index=True)

df_meta.to_csv(home+"data/preprocessed/SDR_metadata.csv", index=False)





###### CHECK THE TECHNICAL BOUNDS FILE BY HAND AFTER RUNNING THIS SCRIPT !!!




































































































































