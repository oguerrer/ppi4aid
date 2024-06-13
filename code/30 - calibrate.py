import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# from joblib import Parallel, delayed
from scipy import signal

warnings.simplefilter("ignore")

home =  os.getcwd()[:-4]




os.chdir(home+'/code/')
import ppi
from functions import *






df = pd.read_csv(home+"data/modeling/indicators_SDR_sample.csv")
groups_dict = dict(zip(df.countryCode, df.group))

shift = 1
colYears = (np.array([year for year in range(1999, 2013)]) + shift).astype(str)




parallel_processes = 50

num_years = len(colYears)
min_value = 1e-2

sub_periods = 4
T = len(colYears)*sub_periods

countries = df.countryCode.unique()




for country in countries:

    # Extract country data
    dft = df[df.countryCode==country]
    df_exp = pd.read_csv(home+"data/modeling/budgets/expenditure/"+country+".csv")
    df_aid = pd.read_csv(home+"data/modeling/budgets/aid/"+country+".csv")
    
    # Indicators
    series = dft[colYears].values
    N = len(dft)
    
    Imax = None
    Imin=None
    R = dft.instrumental.values.copy()
    n = R.sum()
    I0 = series[:,0]
    IF = []
    for serie in series:
        x = np.array([float(year) for year in colYears]).reshape((-1, 1))
        y = serie
        model = LinearRegression().fit(x, y)
        coef = model.coef_
        if coef > 0 and serie[-1] > serie[0]:
            IF.append(serie[-1])
        elif coef > 0 and serie[-1] <= serie[0]:
            IF.append(np.max(serie[serie!=serie[0]]))
        elif coef < 0 and serie[-1] < serie[0]:
            IF.append(serie[-1])
        elif coef < 0 and serie[-1] >= serie[0]:
            IF.append(np.min(serie[serie!=serie[0]]))
    
    IF = np.array(IF)
    success_rates = get_success_rates(series)
    mean_drops = np.array([serie[serie<0].mean() for serie in np.diff(series, axis=1)])
    mean_drops[np.isnan(mean_drops)] = 0
    aa = np.abs(mean_drops/sub_periods)
    
    # Budget 
    expenditure = np.clip([signal.detrend(serie)+np.mean(serie) for serie in df_exp[colYears].values], a_min=0, a_max=None)
    aid = np.clip([signal.detrend(serie)+np.mean(serie) for serie in df_aid[colYears].values], a_min=0, a_max=None)
    Bs = expenditure + aid
    usdgs = sorted(dft.sdg.unique())
    sdg2index = dict(zip(usdgs, range(len(usdgs))))
    sdgs = dft.sdg.values
    B_dict = dict([(i,[sdg2index[sdgs[i]]]) for i in range(N) if R[i]==1])
    Bs = get_dirsbursement_schedule(Bs, B_dict, T)
    
    # Network
    A = np.loadtxt(home+"data/modeling/networks_sdr/"+country+".csv", delimiter=',')
    
    # Governance
    qm = np.ones(n)*dft.controlOfCorruption.values[0]
    rl = np.ones(n)*dft.ruleOfLaw.values[0]
    
    # Perform calibration
    params = np.ones(3*N)*.5
    increment = 1000
    sample_size = 10
    counter = 0
    
    GoF_alpha = np.zeros(N)
    GoF_beta = np.zeros(N)
    
    while np.sum(GoF_alpha<.9) > 0 or np.sum(GoF_beta<.9) > 0 or counter<101:
    
        counter += 1
        alphas = params[0:N]
        alphas_prime = params[N:2*N]
        betas = params[2*N::]

        errors_all = np.array(compute_error(I0, alphas, alphas_prime, betas, A=A, 
                                            R=R, qm=qm, rl=rl, Imax=Imax, Imin=Imin, 
                                            Bs=Bs, B_dict=B_dict, T=T, IF=IF, 
                                            success_rates=success_rates, 
                                            parallel_processes=parallel_processes, 
                                            sample_size=sample_size))

        errors_alpha = errors_all[0:N]
        errors_beta = errors_all[N::]
        
        abs_errors_alpha = np.abs(errors_alpha)
        gaps = IF-I0
        normed_errors_alpha = abs_errors_alpha/np.abs(gaps)
        abs_normed_errors_alpha = np.abs(normed_errors_alpha)
            
        abs_errors_beta = np.abs(errors_beta)
        normed_errors_beta = abs_errors_beta/success_rates
        abs_normed_errrors_beta = np.abs(normed_errors_beta)
        
        params[0:N][(errors_alpha<0) & (IF>I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha<0) & (IF>I0)], .25, .99)
        params[0:N][(errors_alpha>0) & (IF>I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha>0) & (IF>I0)], 1.01, 1.5)
        params[N:2*N][(errors_alpha<0) & (IF>I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha<0) & (IF>I0)], 1.01, 1.5)
        params[N:2*N][(errors_alpha>0) & (IF>I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha>0) & (IF>I0)], .25, .99)
        
        params[0:N][(errors_alpha>0) & (IF<I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha>0) & (IF<I0)], 1.01, 1.5)
        params[0:N][(errors_alpha<0) & (IF<I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha<0) & (IF<I0)], .25, .99)
        params[N:2*N][(errors_alpha>0) & (IF<I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha>0) & (IF<I0)], .25, .99)
        params[N:2*N][(errors_alpha<0) & (IF<I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha<0) & (IF<I0)], 1.01, 1.5)
        
        params[2*N::][errors_beta<0] *= np.clip(1-abs_normed_errrors_beta[errors_beta<0], .25, .99)
        params[2*N::][errors_beta>0] *= np.clip(1+abs_normed_errrors_beta[errors_beta>0], 1.01, 1.5)
        
        
        GoF_alpha = 1 - normed_errors_alpha
        GoF_beta = 1 - abs_normed_errrors_beta
        
        if counter >= 100:
            sample_size += increment
            # increment += 1000
        
        print(country, sample_size, counter,  GoF_alpha.mean(), GoF_beta.mean(), np.min(GoF_alpha.tolist()+GoF_beta.tolist()))
    
    print('computing final estimate...', sample_size)
    print()
    # sample_size = 1000
    alphas_est = params[0:N]
    alphas_prime_est = params[N:2*N]
    betas_est = params[2*N::]
    errors_est = np.array(compute_error(I0, alphas_est, alphas_prime_est, betas_est, 
                                        A=A, R=R, qm=qm, rl=rl, Imax=Imax, Imin=Imin, 
                                        Bs=Bs, B_dict=B_dict, T=T, IF=IF, 
                                        success_rates=success_rates, 
                                        parallel_processes=parallel_processes, 
                                        sample_size=sample_size))
    errors_alpha = errors_est[0:N]
    errors_beta = errors_est[N::]
    
    abs_errors_alpha = np.abs(errors_alpha)
    gaps = IF-I0
    normed_errors_alpha = abs_errors_alpha/np.abs(IF-I0)
    abs_normed_errors_alpha = np.abs(normed_errors_alpha)
        
    abs_errors_beta = np.abs(errors_beta)
    normed_errors_beta = abs_errors_beta/success_rates
    abs_normed_errrors_beta = np.abs(normed_errors_beta)
    
    GoF_alpha = 1 - normed_errors_alpha
    GoF_beta = 1 - abs_normed_errrors_beta
    
    dfc = pd.DataFrame([[alphas_est[i], alphas_prime_est[i], betas_est[i], T, num_years, errors_alpha[i], errors_beta[i], min_value, GoF_alpha[i], GoF_beta[i]] \
                        if i==0 else [alphas_est[i], alphas_prime_est[i], betas_est[i], np.nan, np.nan, errors_alpha[i], errors_beta[i],  np.nan, GoF_alpha[i], GoF_beta[i]] \
                        for i in range(N)], 
                        columns=['alpha', 'alpha_prime', 'beta', 'T', 'years', 'error_alpha', 'error_beta', 'min_value', 'GoF_alpha', 'GoF_beta'])
    dfc.to_csv(home+'data/modeling/parameters_'+str(shift)+'/'+country+'.csv', index=False)
    
    
    
    



















































