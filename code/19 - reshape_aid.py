import matplotlib.pyplot as plt
import numpy as np
import os, copy
import pandas as pd

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/raw/aid_sdgs/FinancingtotheSDGsDataset_v1.0.csv")
dfa = pd.read_csv(home+"data/raw/AidDataCore_v3.0/AidDataCoreFull_ResearchRelease_Level1_v3.0.csv", low_memory=False)
country_codes = dict(zip(dfa.recipient, dfa.recipient_iso))
df['recipient_iso2'] = [country_codes[c] for c in df.recipient.values]
dfn = df[df.recipient_iso2.values.astype(str) != 'nan']


df_iso = pd.read_csv(home+"data/raw/iso_codes.csv")
iso2t3 = dict(zip(df_iso['alpha-2'], df_iso['alpha-3']))

dfn['recipient_iso3'] = [iso2t3[c] if c in iso2t3 else np.nan for c in dfn['recipient_iso2'].values]

goals = ['goal_'+str(i) for i in range(1,18)]


new_rows = []
for country, group in dfn.groupby('recipient_iso3'):
    yeargroups = group.groupby('year').sum()
    for goal in goals:
        new_row = [country, goal.split('_')[-1]] + yeargroups[goal].values.tolist()
        new_rows.append(new_row)
    


dff = pd.DataFrame(new_rows, columns=['countryCode', 'sdg']+sorted(dfn.year.unique()))

dff.to_csv(home+'data/preprocessed/aid.csv', index=False)




# sum_finman = 0
# for index, row in df.iterrows():
#     text = row.long_description
#     if text is not np.nan and row.goal_16 > 0 and ('financ' in text.lower() or 'debt' in text.lower()):
#         print(text.lower())
#         print()
#         sum_finman+=row.goal_16

# print(100*sum_finman/df.goal_16.sum())










































