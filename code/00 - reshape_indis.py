import os, warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_excel(home+"data/raw/indicators/SDR 2021 - Database.xlsx", sheet_name='Data for Trends')
df_gdpg = pd.read_excel(home+"data/raw/indicators/gdp_growth.xls", sheet_name='Data', skiprows=3)


new_rows = []
for country, group in df.groupby('id'):
    data_columns = group.columns.values[3::]
    for column in data_columns:
        column_new = column.replace(' ', '')
        if column_new == 'sdg2_stuntihme':
            column_new = 'sdg2_stunting'
        if column_new == 'sdg2_wasteihme':
            column_new = 'sdg2_wasting'
        new_row = [group.id.values[0], group.Country.values[0], column_new] + group[column].values.tolist()
        new_rows.append(new_row)

all_years = sorted(df.Year.unique())


## Add GDP indicator
for index, row in df_gdpg.iterrows():
    new_row = [row['Country Code'], row['Country Name'], 'sdg8_gdpgrowth']+row[[str(year) for year in all_years[0:-1]]].values.tolist()+[np.nan]
    new_rows.append(new_row)





dfn = pd.DataFrame(new_rows, columns=['countryCode', 'countryName', 'seriesCode']+[str(year) for year in all_years])
dfn.to_csv(home+"data/preprocessed/SDR_reshaped.csv", index=False)





































