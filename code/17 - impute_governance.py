import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/governance_cc_normalized.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)



new_rows = []
for index, row in df.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
        
    if len(observations) < len(colYears) and len(observations) > 5:
        # print(44)
        vals = row[colYears].values.copy()
    
        x = years[observations]
        y = vals[observations]
        X = x.reshape(-1, 1)
    
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        x_pred = years.reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[missing_values] = y_pred[missing_values]
        new_row[years_indices] = vals
        new_rows.append(new_row)
        
    elif len(observations) > 0:
        new_row[years_indices] = row[colYears].values[observations].mean()
        new_rows.append(new_row)
    
dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+"data/preprocessed/governance_cc_imputed.csv", index=False)





df = pd.read_csv(home+"data/preprocessed/governance_rl_normalized.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)



new_rows = []
for index, row in df.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
        
    if len(observations) < len(colYears) and len(observations) > 5:
        # print(44)
        vals = row[colYears].values.copy()
    
        x = years[observations]
        y = vals[observations]
        X = x.reshape(-1, 1)
    
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        x_pred = years.reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[missing_values] = y_pred[missing_values]
        new_row[years_indices] = vals
        new_rows.append(new_row)
        
    elif len(observations) > 0:
        new_row[years_indices] = row[colYears].values[observations].mean()
        new_rows.append(new_row)
    
dff = pd.DataFrame(new_rows, columns=df.columns)
dff.to_csv(home+"data/preprocessed/governance_rl_imputed.csv", index=False)




















































