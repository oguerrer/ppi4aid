import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd
from linear_imputation import impute, Imputer

home =  os.getcwd()[:-4]



df_gdp = pd.read_csv(home+'data/preprocessed/gdp_imputed.csv')
colYears = [c for c in df_gdp.columns if c.isnumeric()]
df_def = pd.read_csv(home+'data/preprocessed/deflator_imputed.csv')







count_gdp = {}
for index, row in df_gdp.iterrows():
    count_gdp[row['Country Code']] = sum(~row[colYears].isnull())

count_inf = {}
for index, row in df_def.iterrows():
    count_inf[row['Country Code']] = sum(~row[colYears].isnull())


able = sorted(list(set(df_gdp['Country Code']).intersection(df_def['Country Code'])))



i2011 = np.where(np.array(colYears)=='2011')[0][0]

new_rows = []
for country in able:
    
    dftx = df_gdp[df_gdp['Country Code'] == country]
    valsx = dftx[colYears].values[0]
    nulls = sum(np.isnan(valsx))
    if nulls > 0:
        temp_df = pd.DataFrame({'years':np.array(colYears).astype(float), 'vals':valsx})
        imp_df = impute(temp_df)
        valsx = imp_df.vals.values
        
    dfti = df_def[df_def['Country Code'] == country]
    valsi = dfti[colYears].values[0]
    deflator = valsi/valsi[i2011]

    vals = valsx / deflator
    new_rows.append( [country] + vals.tolist() )
    
    # plt.plot(vals[-21::])
    # plt.title(country)
    # plt.show()
    

dff = pd.DataFrame(new_rows, columns=['countryCode']+colYears)
dff.to_csv(home+'data/preprocessed/real_gdp.csv', index=False)





























































