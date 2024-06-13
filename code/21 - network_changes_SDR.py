import matplotlib.pyplot as plt
import os, warnings, pycountry
import pandas as pd
import numpy as np


warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/modeling/indicators_SDR_sample.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]



for country, group in df.groupby('countryCode'):
    M = group[colYears].values
    changes = M[:,1::] - M[:,0:-1]
    np.savetxt(home+"data/preprocessed/networks_changes_SDR/"+country+".csv", changes, delimiter=',')


























































