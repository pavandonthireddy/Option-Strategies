# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 20:56:42 2020

@author: Pavan
"""

import numpy as np
import pandas as pd
import itertools
from scipy.optimize import shgo

"""
Parameters
"""
amount = 2000
filename = 'position_size_test.csv'

"""
Data Preperation
"""
data = pd.read_csv(filename)

cost = data['Cost of Strategy'].values
profit = data['Max_Profit'].values
loss = -1*data['Max_Loss'].values
prob_profit = (data['Prob of Profit'].values)/100
prob_loss = 1-prob_profit  

"""
Bet parameters
"""

count_bets = len(data.index)
total_events = 2**count_bets
events = np.array(list(map(list, itertools.product([0, 1], repeat=count_bets))))



wins   = events==1
losses = events==0


prob_profit_mat = np.tile(prob_profit[np.newaxis,:],(total_events,1))
prob_loss_mat = np.tile(prob_loss[np.newaxis,:],(total_events,1))


prob_matrix         = np.empty((total_events,count_bets))
prob_matrix[wins]   = prob_profit_mat[wins]
prob_matrix[losses] = prob_loss_mat[losses]



"""
Optimal Bet allocation which maximizes expected value

"""

def strategy_cost(x):
    x=np.floor(x)
    total_cost = np.dot(x,cost)
    return total_cost

def alloc_exp_value(x):
    
    x=np.floor(x)
    
    alloc_bool = 1*(x!=0)
    total_alloc = np.sum(alloc_bool==1)
    alloc_events = 2**total_alloc
    filler = total_events/alloc_events
    
    pre_mult = np.zeros(total_events)
    pre_mult[0:total_events:int(filler)] = 1
    pre_mult = np.diag(pre_mult)
    
    post_mult = np.zeros(count_bets)
    post_mult[x[0]!=0] =1 
    post_mult = np.diag(post_mult)
    
    use_matrix = np.ones((total_events, count_bets))
    use_matrix = pre_mult@use_matrix@post_mult
    
    
    
    alloc_mat = np.tile(x,(total_events,1))*use_matrix
    
    
    
    
    prob_matrix_aloc = prob_matrix*use_matrix
    
    profit_mat = alloc_mat*use_matrix*np.tile(profit[np.newaxis,:],(total_events,1))
    loss_mat = alloc_mat*use_matrix*np.tile(loss[np.newaxis,:],(total_events,1))
    
    
    winnings = np.zeros((total_events, count_bets))
    winnings[wins] = profit_mat[wins]
    winnings[losses] = loss_mat[losses]
    winnings = winnings*use_matrix
    
    prob_mat_new = prob_matrix_aloc.copy()
    
    prob_mat_new[alloc_mat==0] = 1
    total_winnings = np.sum(winnings,axis=1)

    
    to_add = (amount*(pre_mult@np.ones((total_events,1))))
    final_wealth = total_winnings+np.squeeze(to_add)
    
    
    prob_events = np.prod(prob_mat_new,axis=1)
    prob_events[prob_events==1]=0
    
    expected_value = np.dot(total_winnings,prob_events)
    expected_wealth = np.dot(final_wealth,prob_events)

    return expected_value,expected_wealth

def obj_func(x):
    exp_value, exp_wealth = alloc_exp_value(x)
    return -1*exp_wealth



def cost_cons(x):
    cost = strategy_cost(x)
    return amount-cost

def num_bets_cons(x):
    x=np.floor(x)
    no_bets = np.sum(1*(x!=0))
    return no_bets-2


bounds = [(0,5), ]*count_bets

cons = ({'type': 'ineq', 'fun': cost_cons},
        {'type': 'ineq', 'fun': num_bets_cons})
res = shgo(obj_func,bounds, n=30, sampling_method='sobol',  options ={'disp':True}, iters=3, constraints=cons)
print(np.floor(res.x))
print("Cost of Optimal Strategy     :", strategy_cost(res.x))
# opt_exp_profit,_ = alloc_exp_value(res.x)
# print("Exp Profit of Optimal allocation :", opt_exp_profit)




