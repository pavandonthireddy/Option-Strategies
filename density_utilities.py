# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 00:00:01 2020

@author: Pavan
"""

import math
import numpy as np
from functools import partial
from american_option_pricing import american_option
from scipy.optimize import shgo
import matplotlib  as mpl
import matplotlib.pyplot as plt
import os
"""
x = [w,F1,sigma1,F2,sigma2]

"""


def calculate_densities(chain):
    
    def objective(chain,x):
        T = chain.Time_to_exp/252
        Rf = chain.Rf
        
        c_total = len(chain.Call_Strike)
        p_total = len(chain.Put_Strike)
        
        w= x[0]
        F1= x[1]
        sigma1 = x[2]
        F2 = x[3]
        sigma2 = x[4]
        
        c_market = (chain.Call_Ask+chain.Call_Bid)/2
        p_market = (chain.Put_Ask+chain.Put_Bid)/2
        
        S1 = F1*math.exp(-1*Rf*T)
        S2 = F2*math.exp(-1*Rf*T)
        c_the =np.array([w*(american_option('c',S1,chain.Call_Dummy_Strike[i],T,Rf,sigma1)[0])\
                + (1-w)*(american_option('c',S2,chain.Call_Dummy_Strike[i],T,Rf,sigma2)[0])\
                for i in range(c_total)])
        
        p_the = np.array([w*(american_option('p',S1,chain.Put_Dummy_Strike[i],T,Rf,sigma1)[0])\
                + (1-w)*(american_option('p',S2,chain.Put_Dummy_Strike[i],T,Rf,sigma2)[0])\
                for i in range(p_total)])
    
        c_sse = np.sum((c_market - c_the) ** 2)
        p_sse = np.sum((p_market - p_the) ** 2)
        
        return c_sse+p_sse
    
    def stock_density(S,sigma,r,T,S_T):
        dens = (1/(S_T*sigma*math.sqrt(2*math.pi*T)))*(math.exp(-0.5*(((math.log(S_T)-math.log(S)-r*T+0.5*T*sigma**2)/(sigma*math.sqrt(T)))**2)))
        return dens
    
    def risk_neutral_density(x,chain,S_T):
        w = x[0]
        F1 = x[1]
        sigma1=x[2]
        F2 = x[3]
        sigma2 = x[4]
        r =chain.Rf
        T =chain.Time_to_exp/252
        S1=F1*math.exp(-r*T)
        S2= F2*math.exp(-r*T)
        dens = w*stock_density(S1,sigma1,r,T,S_T)+(1-w)*stock_density(S2,sigma2,r,T,S_T)
        return dens
    
    def real_world_density(x,chain,gamma,S_T):
        r =chain.Rf
        T =chain.Time_to_exp/252   
        w = x[0]
        F1 = x[1]
        sigma1=x[2]
        F2 = x[3]
        sigma2 = x[4]    
        F1_new = F1*math.exp(gamma*T*sigma1**2)
        F2_new = F2*math.exp(gamma*T*sigma2**2)   
        one_by_w_new = 1+((1-w)/w)*((F2/F1)**gamma)*(math.exp(0.5*T*(gamma**2-gamma)*(sigma2**2-sigma1**2)))
        w_new = 1/one_by_w_new
        S1=F1_new*math.exp(-r*T)
        S2= F2_new*math.exp(-r*T)
        dens = w_new*stock_density(S1,sigma1,r,T,S_T)+(1-w_new)*stock_density(S2,sigma2,r,T,S_T)
        return dens

    res_dict=dict()
    
    fun_to_min = partial(objective, chain)
    

    T = chain.Time_to_exp/252
    Rf = chain.Rf
    S = chain.Stock_Last
    
    F = S*math.exp(Rf*T)
    
    bounds =[(0.0,1.0),(0.5*S,1.5*S),(0.0051,0.999999),(0.5*S,1.5*S),(0.0051,0.99999)]    
    

    eq_cons = {'type': 'eq',
               'fun' : lambda x: np.array([x[0]*(x[1])+(1-x[0])*(x[3])- F])}
    
    ineq_cons = {'type': 'ineq',
               'fun' : lambda x: np.array([x[3]-x[1]])}
    

    cons=(eq_cons,ineq_cons)
    
    res = shgo(fun_to_min, bounds, n=120, iters=5, constraints=cons,options={'disp':True}, sampling_method='sobol')    
    
    print(res.x)
    
    res_dict['Name'] = chain.Name
    res_dict['total_calls'] = chain.Call_total
    res_dict['termination'] = res.success
    res_dict['Prob Bearish']=res.x[0]  
    res_dict['Prob Bullish']=1-res.x[0]
    
    if res.x[0]>0.5:
        res_dict['Direction_Price'] = 'Bearish'
    else:
        res_dict['Direction_Price'] = 'Bullish'
        
    res_dict['F1']=res.x[1]
    res_dict['Stock_Last']= chain.Stock_Last
    res_dict['F2']=res.x[3]
    res_dict['sigma1']=res.x[2]
    res_dict['Impl_Vol']=chain.Stock_Volt
    res_dict['sigma2']=res.x[4]
    
    risk_neutral_dens_par = partial(risk_neutral_density,res.x,chain)
    vec_risk_neutral_dens_par = np.vectorize(risk_neutral_dens_par)
    
    real_world_dens_par_1 = partial(real_world_density,res.x,chain,2)
    vec_real_world_dens_par_1 = np.vectorize(real_world_dens_par_1)
    
    real_world_dens_par_2 = partial(real_world_density,res.x,chain,4)
    vec_real_world_dens_par_2 = np.vectorize(real_world_dens_par_2)
    

    
    prices = np.arange(S-0.5*S,S+0.5*S,0.1)
    
    risk_neutral =vec_risk_neutral_dens_par(prices)
    real_world_1 = vec_real_world_dens_par_1(prices)
    real_world_2 = vec_real_world_dens_par_2(prices)
    
    return res_dict, prices, risk_neutral, real_world_1, real_world_2
     

def plot_densities(last,name,path, prices,risk_neutral, real_world_1, real_world_2):
    fig,ax = plt.subplots(1, 1, figsize=(15, 10))
    plt.plot(prices,risk_neutral,lw=2.5, color='blue', label = "Risk Neutral Density $S_T$")
    plt.plot(prices,real_world_1,lw=2.5, color='red', label="Real World Density $S_T, \gamma=2 $")
    plt.plot(prices,real_world_2,lw=2.5, color='orange', label="Real World Density $S_T, \gamma=4$")
    
    plt.axvline(x = last, color="green", linestyle='--', label = "$S_0 : {}$".format(last)) 
    plt.xlabel('$S_T$')
    plt.ylabel('Density')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    plt.legend()
    plt.grid(True)
    file_name = name+".png"
    plt.savefig(os.path.join(path, file_name))
    plt.close(fig)