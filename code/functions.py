# -*- coding: utf-8 -*-
"""Economic Complexity and Sustainable Development: 
    A Computational Framework for Policy Priorities

Helper functions

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Pyhton 3.7


"""
import os
import numpy as np
from joblib import Parallel, delayed


home =  os.getcwd()[:-4]
os.chdir(home+'/code/')
import ppi



# def get_conditions(series, min_value, Imax=None):
#     I0 = series[:,0]
#     IF = series[:,-1]
#     gaps = IF - I0
#     IF[gaps < min_value] = I0[gaps < min_value] + min_value
#     if Imax is not None:
#         invalid = IF >= Imax
#         IF[invalid] = Imax[invalid]
#         I0[invalid] = IF[invalid]-min_value
#     return I0, IF



def get_success_rates(series):
    sc = series[:, 1::]-series[:, 0:-1] # get changes in indicators
    success_rates = np.sum(sc>0, axis=1)/sc.shape[1] # compute rates of success
    success_rates = .9*(success_rates-success_rates.min())/(success_rates.max()-success_rates.min()) + .05
    return success_rates



def get_dirsbursement_schedule(Bs, B_dict, T):
    programs = sorted(list(set([item for subl in list(B_dict.values()) for item in subl])))
    B_sequence = [[] for program in programs]
    subperiods = int(T/Bs.shape[1])
    for i, program in enumerate(programs):
        for period in range(Bs.shape[1]):
            for subperiod in range(subperiods):
                B_sequence[i].append( Bs[i,period]/subperiods )
    B_sequence = np.array(B_sequence)
    return B_sequence




def run_ppi_parallel(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None,
            Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=50, frontier=None):
    outputs = ppi.run_ppi(I0, alphas, alphas_prime, betas, A=A, R=R, bs=bs, qm=qm, rl=rl,
            Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict, G=G, T=T, frontier=frontier)
    return outputs




def compute_error(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None,
            Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=50, frontier=None, 
            IF=None, success_rates=None, parallel_processes=1, sample_size=30):
    sols = Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi_parallel)\
            (I0, alphas, alphas_prime, betas, A=A, R=R, bs=bs, qm=qm, rl=rl,
            Imax=Imax, Imin=Imin, Bs=Bs, B_dict=B_dict, G=G, T=T, frontier=frontier) for itera in range(sample_size))
    
        
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    I_hat = np.mean(tsI, axis=0)[:,-1]
    gamma_hat = np.mean(tsG, axis=0).mean(axis=1)
    error_alpha = IF - I_hat
    error_beta = success_rates - gamma_hat
    return error_alpha.tolist() + error_beta.tolist()








































































