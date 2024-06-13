import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


dfcc = pd.read_excel(home+"data/raw/governance/wgidataset.xlsx", sheet_name='ControlofCorruption', skiprows=14)
dfrl = pd.read_excel(home+"data/raw/governance/wgidataset.xlsx", sheet_name='RuleofLaw', skiprows=14)
df_indis = pd.read_csv(home+"data/preprocessed/SDR_corrected.csv")


colYears = [str(c) for c in range(1996, 2001, 2)]+[str(c) for c in range(2002, 2020)]


relevant_columns = [c for c in dfcc.columns if 'Estimate' in c or 'Code' in c]
dfcc_clean = pd.DataFrame(dfcc[relevant_columns].values, columns=['countryCode']+colYears)
dfrl_clean = pd.DataFrame(dfrl[relevant_columns].values, columns=['countryCode']+colYears)


dffc = dfcc_clean[dfcc_clean.countryCode.isin(df_indis.countryCode)]
dffr = dfrl_clean[dfrl_clean.countryCode.isin(df_indis.countryCode)]


dffc.to_csv(home+"data/preprocessed/governance_cc_reshaped.csv", index=False)
dffr.to_csv(home+"data/preprocessed/governance_rl_reshaped.csv", index=False)


















































