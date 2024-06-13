import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from scipy import signal

warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]




os.chdir(home+'/code/')
import ppi
from functions import *



parallel_processes = 50
sample_size = 1000


df = pd.read_csv(home+"data/modeling/indicators_SDR_sample.csv")
groups_dict = dict(zip(df.countryCode, df.group))

shift = 1
colYears = (np.array([year for year in range(1999, 2013)]) + shift).astype(str)



countries = df.countryCode.unique()

for country in countries:

    dft = df[df.countryCode==country]
    df_exp = pd.read_csv(home+"data/modeling/budgets/expenditure/"+country+".csv")
    df_aid = pd.read_csv(home+"data/modeling/budgets/aid/"+country+".csv")
    df_params = pd.read_csv(home+"data/modeling/parameters_"+str(shift)+"/"+country+".csv")
    
    # Parameters
    alphas = df_params.alpha.values
    alphas_prime = df_params.alpha_prime.values
    betas = df_params.beta.values
    T = int(df_params['T'].values[0])
    num_years = int(df_params.years.values[0])
    sub_periods = int(T/num_years)
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    R = dft.instrumental.values.copy()
    n = R.sum()
    I0 = series[:,0]
    Imax = np.ones(N)
    Imin = np.zeros(N)
    
    # Budget 
    expenditure = np.clip([signal.detrend(serie)+np.mean(serie) for serie in df_exp[colYears].values], a_min=0, a_max=None)
    aid = np.clip([signal.detrend(serie)+np.mean(serie) for serie in df_aid[colYears].values], a_min=0, a_max=None)
    usdgs = sorted(dft.sdg.unique())
    sdg2index = dict(zip(usdgs, range(len(usdgs))))
    sdgs = dft.sdg.values
    B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
    
    # Network
    A = np.loadtxt(home+"data/modeling/networks_SDR/"+country+".csv", delimiter=',')
    
    # Governance
    qm = np.ones(n)*dft.controlOfCorruption.values[0]
    rl = np.ones(n)*dft.ruleOfLaw.values[0]
    
    
    # Benchmark
    print(country, 'benchmark...')
    Bs = get_dirsbursement_schedule(expenditure+aid, B_dict, T)
    sols = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl, Imax=Imax, 
             Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(sample_size))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_indis_0 = np.mean(tsI, axis=0)
    
    
    # Remove all aid
    print(country, 'impact...')
    Bs = get_dirsbursement_schedule(expenditure, B_dict, T)
    sols = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, alphas, alphas_prime, betas, A=A, R=R, qm=qm, rl=rl, Imax=Imax, 
             Imin=Imin, Bs=Bs, B_dict=B_dict) for itera in range(sample_size))
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    aver_indis_c = np.mean(tsI, axis=0)
        
    M = np.column_stack( [dft.seriesCode.values, aver_indis_0, aver_indis_c] )
    df_effects = pd.DataFrame(M, columns=['seriesCode'] + ['baseline_'+str(c) for c in range(T)] + ['counter_'+str(c) for c in range(T)])
    df_effects.to_csv(home+'data/modeling/effects_'+str(shift)+'/'+country+'.csv', index=False)
    
    
    
    
    




















