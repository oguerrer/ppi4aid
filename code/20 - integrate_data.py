import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


df_aid = pd.read_csv(home+'data/preprocessed/aid.csv')

country_sample = df_aid.countryCode.unique()


df_indis = pd.read_csv(home+"data/preprocessed/SDR_corrected.csv")
colYears = [col for col in df_indis.columns if str(col).isnumeric() and int(col)<2021]
df_indis = df_indis[df_indis[colYears].isnull().sum(axis=1) == 0]

df_groups = pd.read_excel(home+"data/raw/geography/country_groups.xlsx")
groups_dict = dict(zip(df_groups.countryCode, df_groups.region))
df_indis = df_indis[df_indis.countryCode.isin(list(groups_dict.keys()))]

M = df_indis[colYears].values
changes = np.abs(M[:,1::]-M[:,0:-1])
dynamic = np.sum(changes, axis=1) > 10e-6
df_indis = df_indis[dynamic]


sdg_coverage = []
for country, group in df_indis.groupby('countryCode'):
    if len(group.sdg.unique()) >= 1: # have indicators for at least 10 SDGs
        sdg_coverage.append(country)

df_indis = df_indis[df_indis.countryCode.isin(sdg_coverage)]



df_expen = pd.read_csv(home+"data/preprocessed/expenditure_percapita.csv")
df_gov_cc = pd.read_csv(home+"data/preprocessed/governance_cc_corrected.csv")
df_gov_rl = pd.read_csv(home+"data/preprocessed/governance_rl_corrected.csv")



common_countries = set(df_indis.countryCode.values).intersection(df_expen['Country Code'].values).intersection(df_gov_cc.countryCode.values).intersection(df_gov_rl.countryCode.values)
common_countries = common_countries.intersection(country_sample)

dfi = df_indis[df_indis.countryCode.isin(common_countries)]
dfx = df_expen[df_expen['Country Code'].isin(common_countries)]
dfcc = df_gov_cc[df_gov_cc.countryCode.isin(common_countries)]
dfrl = df_gov_rl[df_gov_rl.countryCode.isin(common_countries)]
df_aid = df_aid[df_aid.countryCode.isin(common_countries)]
aidYears = df_aid.columns[2::]

# Turn aid data into per capita
df_pop = pd.read_csv(home+"data/preprocessed/population_reshaped.csv")
pop_dict = dict([(row.countryCode, row[aidYears].values) for index, row in df_pop.iterrows()])

new_rows = []
for index, row in df_aid.iterrows():
    new_row = [row.countryCode, row.sdg] + (row[aidYears]/pop_dict[row.countryCode]).tolist()
    new_rows.append(new_row)

df_aid_final = pd.DataFrame(new_rows, columns = df_aid.columns)
df_aid_final.to_csv(home+"data/modeling/aid_data.csv", index=False)



cc_dict = dict(zip(dfcc.countryCode, dfcc.values[:,1::].mean(axis=1)))
rl_dict = dict(zip(dfrl.countryCode, dfrl.values[:,1::].mean(axis=1)))


dfi['controlOfCorruption'] = [cc_dict[country] for country in dfi.countryCode.values]
dfi['ruleOfLaw'] = [rl_dict[country] for country in dfi.countryCode.values]


dfi['group'] = [groups_dict[country] for country in dfi.countryCode.values]

# For this cample, change the countries in Oceania to the 'East & South Asia' group
dfi.loc[dfi.group=='Oceania', 'group'] = 'East & South Asia'


# Add column identifying instrumental indicators
dfm = pd.read_csv(home+"data/preprocessed/SDR_tech_bounds.csv")
instrumentals = dict(zip(dfm.IndCode, dfm.instrumental))
dfi['instrumental'] = [instrumentals[indi] for indi in dfi.seriesCode.values]

# Remove corruption index
dfc = dfi[dfi.seriesCode == 'sdg16_cpi']
dfc.to_csv(home+"data/modeling/corruption_index.csv", index=False)
dfi = dfi[dfi.seriesCode != 'sdg16_cpi']

# Sort rows
dfi.sort_values(by=['countryCode', 'sdg', 'seriesCode'], inplace=True)

# Sort columns
scolumns = sorted(dfi.columns.values)
dfif = pd.DataFrame(dfi[scolumns].values, columns=scolumns)
dfif = dfif[dfif.seriesCode != 'sdg17_oda']
variation = dfif[colYears[0:13]].nunique(axis=1)
dfif = dfif[variation > 1]
dfif.to_csv(home+"data/modeling/indicators_SDR_sample.csv", index=False)

# validation dataset
dfval = dfif[[col for col in dfif.columns if not col.isnumeric() or (int(col)>=2002) and int(col)<=2012]]
dfval.to_csv(home+"data/modeling/indicators_SDR_validation.csv", index=False)


dfx_final = pd.DataFrame(dfx.values, columns=['countryCode']+dfx.columns.tolist()[1::])
dfx_final.to_csv(home+"data/modeling/expenditure_wb_sample.csv", index=False)

dfx_val = dfx_final[[col for col in dfx_final.columns if not col.isnumeric() or (int(col)>=2002) and int(col)<=2012]]
dfx_val.to_csv(home+"data/modeling/expenditure_wb_validation.csv", index=False)


for country in dfx_final.countryCode.values:
    df_exp_c = pd.read_csv(home+"data/preprocessed/expenditure_synthetic_per_capita/"+country+".csv")
    df_exp_c.to_csv(home+"data/modeling/expenditure_synthetic/"+country+".csv", index=False)

    df_exp_val = df_exp_c[[col for col in df_exp_c.columns if not col.isnumeric() or (int(col)>=2002) and int(col)<=2012]]
    df_exp_val.to_csv(home+"data/modeling/expenditure_synthetic_validation/"+country+".csv", index=False)






























































































