import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/expenditure_percapita.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)



# df = df[df['Country Code']=='GNQ']
for index, row in df.iterrows():
    
    country = row['Country Code']
    print(country)    
    
    new_rows = []
    new_row = row.values.copy()
    
    
        
    vals0 = row[colYears].values.copy()
    min_val = np.nanmin(vals0)
    max_val = np.nanmax(vals0)
    vals0 = (vals0 - min_val)/(max_val - min_val)

    x = years
    y = vals0
    X = x.reshape(-1, 1)

    kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)

    x_pred = years.reshape(-1,1)
    y_pred, sigma = gp.predict(x_pred, return_std=True)

    # synthetic_series = []
    # seed = 10
    # while len(synthetic_series) < 1000:
    #     synthetic_serie = gp.sample_y([[year] for year in years], n_samples=1, random_state=seed).T[0]
    #     seed+=1
    #     if sum(synthetic_serie[0:14]) < sum(y[0:14]):
    #         synthetic_series.append(synthetic_serie)
    synthetic_series = gp.sample_y([[year] for year in years], n_samples=1000).T
    
    if sum(y > 1):
        print(1, country)

    for series_num, valss in enumerate(synthetic_series):           
    
        print(country, series_num)
        new_row = row.values.copy()
        vals = valss.copy()
        vals = vals*(max_val - min_val) + min_val
        vals[vals<0] = min_val
        
        
        new_row[years_indices] = vals
        new_rows.append(new_row)
        
    
    
        dff = pd.DataFrame(new_rows, columns=df.columns)    
        dff.to_csv(home+"data/preprocessed/expenditure_synthetic_per_capita/"+country+".csv", index=False)
    
        
        
        
        
    