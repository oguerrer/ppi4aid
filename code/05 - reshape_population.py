import matplotlib.pyplot as plt
import os, warnings
import pandas as pd
import numpy as np
warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]


df = pd.read_csv(home+"data/preprocessed/SDR_reshaped.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]

dff = df[df.seriesCode == 'Population']
dff.reset_index(inplace=True)
dff.to_csv(home+"data/preprocessed/population_reshaped.csv", index=False)












































































































































