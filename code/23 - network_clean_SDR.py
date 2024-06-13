import matplotlib.pyplot as plt
import os, warnings, pycountry
import pandas as pd
import numpy as np


warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]



df = pd.read_csv(home+"data/modeling/indicators_SDR_sample.csv")
colYears = [col for col in df.columns if str(col).isnumeric()]



all_links = []
for country in df.countryCode.unique():
    A = np.loadtxt(home+'/data/preprocessed/networks_sparse_SDR/'+country+'.csv', dtype=float, delimiter=" ")  
    links = A.flatten()
    links = links[links!=0]
    all_links += links.tolist()


perb = np.percentile(all_links, 5)
pert = np.percentile(all_links, 95)





for country, group in df.groupby('countryCode'):

    SDGs = group.sdg.values
    
    A = np.loadtxt(home+'/data/preprocessed/networks_sparse_SDR/'+country+'.csv', dtype=float, delimiter=" ")  

    links = A.flatten()
    links = links[links!=0]
    A[A>pert] = 0
    A[A<perb] = 0
    
    dfc = df[df.countryCode==country]
    indi2ix = dict(zip(dfc.seriesCode.values, range(len(dfc))))
    

    for index, row in group.iterrows():
    
        indio = row.seriesCode
                        
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if SDGs[i] == SDGs[j] and A[i,j] < 0:
                A[i,j] = 0
                                        

    np.savetxt(home+"data/modeling/networks_SDR/"+country+".csv", A, delimiter=',')






























































































