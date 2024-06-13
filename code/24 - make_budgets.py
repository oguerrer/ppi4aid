import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


dfa = pd.read_csv(home+"data/modeling/aid_data.csv")
dfa = dfa[dfa.sdg != 12]
colYears = [col for col in dfa.columns if str(col).isnumeric()]

dfi = pd.read_csv(home+"data/modeling/indicators_SDR_sample.csv")
groups_dict = dict(zip(dfi.countryCode, dfi.group))

aid_dict = {}
for country, group in dfa.groupby('countryCode'):
    aid_dict[country] = np.nansum(group[colYears].values)


dfx = pd.read_csv(home+"data/modeling/expenditure_wb_sample.csv")




total_aid = np.nansum(dfa[colYears].values)
fractions = {}
for sdg, group in dfa.groupby('sdg'):
    fractions[sdg] = np.nansum(group[colYears].values)/total_aid




for country in dfx.countryCode.values:
    
    dfit = dfi[dfi.countryCode==country]
    dfxt = dfx[dfx.countryCode==country]
    dfat = dfa[dfa.countryCode==country]

    N = len(dfit)
    R = dfit.instrumental.values.copy()
    usdgs = sorted(dfit.sdg.unique())
    sdg2index = dict(zip(usdgs, range(len(usdgs))))
    sdgs = dfit.sdg.values
    B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])

    aid_inflows = dict([(row.sdg, row[colYears].values) for index, row in dfat.iterrows()])

    Bs = []
    Bs_aid = []
    Bs_exp = []
    for sdg in usdgs:
        aid_series = aid_inflows[sdg].astype(float)
        aid_series[np.isnan(aid_series)] = 0
        fraction = fractions[sdg]
        expenditure_series = fraction*dfxt[colYears].values[0]
        budget_series = expenditure_series + aid_series
        Bs.append([sdg]+budget_series.tolist())
        Bs_aid.append([sdg]+aid_series.tolist())
        Bs_exp.append([sdg]+expenditure_series.tolist())

    df_Bs = pd.DataFrame(Bs, columns=['sdg']+colYears)
    df_Bs_aid = pd.DataFrame(Bs_aid, columns=['sdg']+colYears)
    df_Bs_exp = pd.DataFrame(Bs_exp, columns=['sdg']+colYears)

    df_Bs.to_csv(home+"data/modeling/budgets/total/"+country+".csv", index=False)
    df_Bs_aid.to_csv(home+"data/modeling/budgets/aid/"+country+".csv", index=False)
    df_Bs_exp.to_csv(home+"data/modeling/budgets/expenditure/"+country+".csv", index=False)




















