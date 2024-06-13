import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/SDR_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]

dfm = pd.read_csv(home+"data/preprocessed/SDR_tech_bounds.csv")
dfm.set_index(dfm.IndCode, inplace=True)


cols = df.columns
new_rows = []
goals = []
sdgs = []
pop_data = []
for index, row in df.iterrows():
    new_row = row.values
    if row.seriesCode != 'Population' and row.seriesCode in dfm.IndCode.values:
        vals = row[colYears].values
        meta_row = dfm.loc[row.seriesCode]
        lb = meta_row.lower_bound
        ub = meta_row.upper_bound
        nvals = (vals-lb)/(ub-lb)
        nvals[nvals<0] = 0
        nvals[nvals>1] = 1
        ngoal = (meta_row['Optimum (= 100)']-lb)/(ub-lb)
        
        if meta_row.invert==1:
            nvals = 1 - nvals
            ngoal = 1 - ngoal
            
        if ngoal > 1:
            print('Normalization error in', row.seriesCode)

        new_row[np.in1d(cols, colYears)] = nvals
        new_rows.append(new_row)
        goals.append(ngoal)
        sdgs.append(meta_row.SDG)
    else:
        pop_data.append( [row.countryCode]+row[colYears].values.tolist() )


dff = pd.DataFrame(new_rows, columns=cols)
dff['goal'] = goals
dff['sdg'] = sdgs


dff.to_csv(home+"data/preprocessed/SDR_normalized.csv", index=False)


df_pop = pd.DataFrame(pop_data, columns=['countryCode']+colYears)
df_pop.to_csv(home+"data/preprocessed/SDR_population.csv", index=False)












































