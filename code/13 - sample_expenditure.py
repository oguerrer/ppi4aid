import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df_exp = pd.read_csv(home+"data/preprocessed/expenditure_imputed.csv")
df_gdp = pd.read_csv(home+"data/preprocessed/real_gdp.csv")

df_pop = pd.read_csv(home+"data/preprocessed/population_reshaped.csv")
colYears = [col for col in df_pop.columns if str(col).isnumeric() and int(col)<2021]


pop_dict = dict([(row.countryCode, row[colYears].values) for index, row in df_pop.iterrows()])

sample_exp = df_exp[df_exp[colYears].isnull().sum(axis=1)==0]
sample_gdp = df_gdp[df_gdp[colYears].isnull().sum(axis=1)==0]

common_countries = sorted(list(set(sample_exp['Country Code'].values).intersection(sample_gdp['countryCode'].values)))

exp_dict = dict([(row['Country Code'], row[colYears].values) for index, row in sample_exp.iterrows()])
gdp_dict = dict([(row['countryCode'], row[colYears].values) for index, row in sample_gdp.iterrows()])


new_rows = []
for country in common_countries:
    if country in pop_dict:
        new_row = [country] + (gdp_dict[country]*(exp_dict[country]/100)/pop_dict[country]).tolist()
        new_rows.append(new_row)
    





dff = pd.DataFrame(new_rows, columns=['Country Code']+colYears)
dff.to_csv(home+"data/preprocessed/expenditure_percapita.csv", index=False)





new_rows = []
for country in common_countries:
    if country in pop_dict:
        new_row = [country] + (gdp_dict[country]*(exp_dict[country]/100)).tolist()
        new_rows.append(new_row)
    





dff = pd.DataFrame(new_rows, columns=['Country Code']+colYears)
dff.to_csv(home+"data/preprocessed/expenditure_total.csv", index=False)







































