import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/preprocessed/gdp_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]
years = np.array([int(col) for col in df.columns if str(col).isnumeric()])
years_indices = df.columns.isin(colYears)

new_rows = []
for index, row in df.iterrows():
    
    observations = np.where(~row[colYears].isnull())[0]
    missing_values = np.where(row[colYears].isnull())[0]
    new_row = row.values.copy()
        
    if len(observations) < len(colYears) and len(observations) > 3:
        
        vals = row[colYears].values.copy()
        min_val = np.nanmin(vals)
        max_val = np.nanmax(vals)
        vals = (vals - min_val)/(max_val - min_val)
    
        x = years[observations]
        y = vals[observations]
        X = x.reshape(-1, 1)
    
        kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)
        gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)

        x_pred = years.reshape(-1,1)
        y_pred, sigma = gp.predict(x_pred, return_std=True)
        
        vals[missing_values] = y_pred[missing_values]
        
        # print(np.nansum(vals) - np.nansum(row[colYears].values))
        
        vals = vals*(max_val - min_val) + min_val
        
        
        ## Correction to avoid unexpected jumps or negative values
        first_observation = observations[0]
        last_observation = observations[-1]
        
        vv = vals[first_observation:last_observation+1]
        vv = np.abs(vv[1::] - vv[0:-1])
        if np.sum(vv==0)==len(vv):
            vv = 10e-12
        
        vvf = vals[0:first_observation+1]
        vvf = np.abs(vvf[1::] - vvf[0:-1])
        
        vvl = vals[last_observation::]
        vvl = np.abs(vvl[1::] - vvl[0:-1])
        
        # check that there are missing values before the first observation
        if len(vvf) > 0:
            
            first_less_than_lb = np.sum(vals[0:first_observation]<0) > 0
            first_bigger_jump = np.max(vvf) > np.max(vv)
            
            if first_bigger_jump or first_less_than_lb:

                if vals[first_observation] == 0:
                    vals[first_observation] = 10e-12
                if vals[first_observation] == 1:
                    vals[first_observation] = 1-10e-12
                ref_val = vals[first_observation]
                
                while first_bigger_jump or first_less_than_lb:
                    
                    diff = 0.999*(vals - ref_val)
                    vals[0:first_observation] = ref_val + diff[0:first_observation]
                    vvf = vals[0:first_observation+1]
                    vvf = np.abs(vvf[1::] - vvf[0:-1])
                    
                    first_less_than_lb = np.sum(vals[0:first_observation]<0) > 0
                    first_bigger_jump = np.max(vvf) > np.max(vv)
        
        if len(vvl) > 0:
            
            last_less_than_lb = np.sum(vals[last_observation+1::]<0) > 0
            last_bigger_jump = np.max(vvl) > np.max(vv)

            if last_bigger_jump or last_less_than_lb:

                if vals[last_observation] == 0:
                    vals[last_observation] = 10e-12
                if vals[last_observation] == 1:
                    vals[last_observation] = 1-10e-12
                ref_val = vals[last_observation]

                while last_bigger_jump or last_less_than_lb:
                    
                    diff = 0.999*(vals - ref_val)
                    vals[last_observation+1::] = ref_val + diff[last_observation+1::]
                    vvl = vals[last_observation::]
                    vvl = np.abs(vvl[1::] - vvl[0:-1])
                    
                    last_less_than_lb = np.sum(vals[last_observation+1::]<0) > 0
                    last_bigger_jump = np.max(vvl) > np.max(vv)
        
        
        new_row[years_indices] = vals
        
        if np.sum(vals[40::]<0) > 0:
            print(row['Country Name'])
        
        plt.plot(vals)
        plt.plot(row[colYears].values)
        plt.show()

        new_rows.append(new_row)
        
    elif len(observations) == len(colYears):
        new_rows.append(new_row)
    

dff = pd.DataFrame(new_rows, columns=df.columns)    
dff.to_csv(home+"data/preprocessed/gdp_imputed.csv", index=False)

    
    
    
    
    