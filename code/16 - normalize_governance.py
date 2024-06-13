import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/governance_cc_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
min_val = min([np.nanmin(df[colYears].values), -2.5])
max_val = max([np.nanmax(df[colYears].values), 2.5])

new_rows = []
for index, row in df.iterrows():
    vals = row[colYears].values.copy()
    valsn = (vals - min_val)/(max_val - min_val)
    new_row = [row.countryCode] + valsn.tolist()
    new_rows.append(new_row)

dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+"data/preprocessed/governance_cc_normalized.csv", index=False)








df = pd.read_csv(home+"data/preprocessed/governance_rl_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
min_val = min([np.nanmin(df[colYears].values), -2.5])
max_val = max([np.nanmax(df[colYears].values), 2.5])

new_rows = []
for index, row in df.iterrows():
    vals = row[colYears].values.copy()
    valsn = (vals - min_val)/(max_val - min_val)
    new_row = [row.countryCode] + valsn.tolist()
    new_rows.append(new_row)

dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+"data/preprocessed/governance_rl_normalized.csv", index=False)






















