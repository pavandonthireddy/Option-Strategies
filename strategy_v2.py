
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:51:08 2020

@author: Pavan
"""

import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import math
import matplotlib  as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
mpl.rcParams['font.family'] = 'serif'
import scipy.stats as stats
import itertools
from datetime import datetime
import os


"""
#######################################################################################
                                      Import Data                     
#######################################################################################
"""

data = pd.read_excel('data_v1.xlsx', index_col=None)  
max_quantity_per_leg = 5
min_e_pnl = 0

Strategies = ["Bear Call Spread","Bull Call Spread", \
              "Bull Put Spread", "Bear Put Spread",\
              "Bull Put Ladder", "Bear Call Ladder",\
              "Long Straddle", "Long Strangle", \
              "Short Straddle", "Short Strangle", \
              "Long Call Butterfly", "Long Put Butterfly",\
              "Short Call Butterfly", "Short Put Butterfly",\
              "Long Iron Butterfly", "Short Iron Butterfly",\
              "Long Call Condor", "Long Put Condor", \
              "Short Call Condor", "Short Put Condor", \
              "Long Iron Condor", "Short Iron Condor", \
              "Long Box"\
              ]


"""
#######################################################################################
                                    Data Cleaning                    
#######################################################################################
"""



def wrang_1(df, col_names):
    for col in col_names:
 
        df[col] = df[col].str.rstrip('%')
        df[col] = pd.to_numeric(df[col],errors='coerce')
        df[col] = [float(x)/100.0 for x in df[col].values]
    return df

convert_cols = ["Impl Vol", "Prob.ITM","Prob.OTM","Prob.Touch"]
data = wrang_1(data,convert_cols)

def label_type(row):
   if row['Symbol'][0] == "." :
      return 'Option'
   return 'Stock'
data['Type']=data.apply(lambda row: label_type(row), axis=1)

data['Expiry_Date']= data.Symbol.str.extract('(\d+)')
data['Expiry_Date'] = data['Expiry_Date'].apply(lambda x: pd.to_datetime(str(x), format='%y%m%d'))
data['Expiry_Date'] = data['Expiry_Date'].fillna(pd.Timestamp('20200724'))
data['Expiry_Date'] = data['Expiry_Date'].apply(lambda x: x.strftime('%Y_%m_%d'))

#TODO: Change the logic for new symbol. Works only for this year. 
data['Group']= data.Symbol.apply(lambda st: st[st.find(".")+1:st.find("20")])

data['Group'] = np.where(data['Type'] == "Stock", data['Symbol'],data['Group']) 

data['Chain_ID'] = data['Group']+"_"+data['Expiry_Date']

data['Spread'] = data['Bid']-data['Ask']


stock_data = data[data['Type'] == "Stock"]
stock_data.rename(columns={"Description": "stock_Description", 
                           "Last": "stock_Last",
                           "High":"stock_High",
                           "Low":"stock_Low",
                           "Open" : "stock_Open",
                           "Volume":"stock_Volume",
                           "Bid":"stock_Bid",
                           "Ask":"stock_Ask",
                           "Impl Vol":"stock_Impl_Vol",
                           "Spread":"stock_Spread"}, inplace=True)    
stock_data = stock_data[["stock_Description","stock_Last","stock_High","stock_Low",
                         "stock_Open","stock_Volume","stock_Bid","stock_Ask",
                         "stock_Impl_Vol","stock_Spread","Chain_ID"]]
    
option_data = data[data['Type']=="Option"]
option_data['Option_type'] = option_data.loc[:,'Description'].str.split(' ').str[-1]

final_dataset = pd.merge(option_data,stock_data,on=['Chain_ID'])



"""
#######################################################################################
                                    Option Chain Class                   
#######################################################################################
"""


class Option_chain(object):
    
    def __init__(self,asset,final_dataset):
        self.Name = asset
        asset_df = final_dataset[final_dataset['Group']==asset]
        
        asset_df["Dummy_Strike"] = asset_df.loc[:,"Strike"]
        
        filtered = asset_df.groupby('Strike')['Strike'].filter(lambda x: len(x) == 2)
        not_common = asset_df[~asset_df['Strike'].isin(filtered)]
        records = not_common.to_dict('records')
        to_add = list()
        for record in records:
            new = record.copy()
            if record["Option_type"] == "CALL":
                for keys,value in new.items():
                    if type(value) ==str :
                        if keys in ["Symbol","Description"]:
                            new[keys] = "Dummy"
                        if keys == "Option_type":
                            new[keys] = "PUT"
                    else:
                        if keys[:5] != "stock" and (keys not in ["Strike", "Dummy_Strike"]):
                            new[keys] = 0 
                new["Strike"] = -1000000
                
            else:
                for keys,value in new.items():
                    if type(value) ==str :
                        if keys in ["Symbol","Description"]:
                            new[keys] = "Dummy"
                        if keys == "Option_type":
                            new[keys] = "CALL"
                    else:
                        if keys[:5] != "stock" and (keys not in  ["Strike", "Dummy_Strike"]):
                            new[keys] = 0 
                            
                new["Strike"] = 1000000
        
            to_add.append(new)
            
        df_dummy_options = pd.DataFrame(to_add)
        
        asset_df = asset_df.append(df_dummy_options)
        asset_df = asset_df.sort_values(by=['Option_type', 'Dummy_Strike'])
        
        greeks = ["Delta","Gamma","Theta","Vega","Rho"]
        for greek in greeks:
            setattr(self,"Var_"+greek, np.std(asset_df[greek].values))
        
        
        self.lot_size = 100

        
        self.call_df = asset_df[asset_df['Option_type']=="CALL"]
        self.put_df = asset_df[asset_df['Option_type']=="PUT"]
        
        
        cols = ["Bid","Ask", "Last", "Delta","Gamma","Theta","Vega","Rho","Impl Vol","Strike", "Dummy_Strike"]
        
        for col in cols:
            setattr(self,"Call_"+col,self.call_df[col].values)
            setattr(self,"Put_"+col,self.put_df[col].values)
            

        setattr(self,"Max_strike", np.max(self.Call_Strike))
        setattr(self,"Min_strike", np.min(self.Call_Strike))

        
        setattr(self,"Call_total", len(self.Call_Strike))
        setattr(self,"Put_total", len(self.Put_Strike))
        
        setattr(self,"Min_strike_width", np.min(np.diff(self.Call_Strike)))
        setattr(self,"Max_strike_width", np.max(np.diff(self.Call_Strike)))
        
        setattr(self,"Stock_Last", asset_df["stock_Last"].iloc[0])
        
        setattr(self,"Stock_Volt", asset_df["stock_Impl_Vol"].iloc[0])
        setattr(self,"Time_to_exp", 10)
        
        std = self.Stock_Last*(self.Stock_Volt*((self.Time_to_exp/252)**0.5))
        self.sigma = std
        
        R_year = 0.05 # Risk Free Interest Rate
        
        vol_day = (self.Stock_Volt)*math.sqrt(1/252)
        
        ln_S_0 = math.log(self.Stock_Last)
        rt = R_year*(self.Time_to_exp/252)
        s_term= -0.5*(vol_day**2)*(self.Time_to_exp)
        
        mu_n = ln_S_0+rt+s_term
        sigma_n = vol_day*math.sqrt(self.Time_to_exp)
        
        
        
        s = sigma_n
        
        scale = math.exp(mu_n)
        
        self.lognormal_s = s
        self.lognormal_scale =scale
        
        self.S_space = np.linspace(self.Stock_Last - 4*self.sigma, self.Stock_Last + 4*self.sigma, 20)
        
        self.S_density = stats.lognorm.pdf(self.S_space,s,loc=0,scale=scale)
       
        
        
        
        
        
        
        ## for optimization 
        variables = ["Ask","Bid","Strike","Delta","Gamma","Theta","Vega","Rho"]
        for variable in variables:
            cal_cont = getattr(self,"Call_"+variable)
            put_cont = getattr(self,"Put_"+variable)
            array = np.hstack((cal_cont[:,np.newaxis],put_cont[:,np.newaxis]))
            setattr(self, variable+"_List" , [[ array[i,j] for j in range(array.shape[1])] for i in range(array.shape[0])])
            
        
"""
#######################################################################################
                                    Strategy Class                
#######################################################################################
""" 



class Strategy(object):
    
    def __init__(self,allocation,chain,name):
        self.Option_Chain = chain
        self.Call_allocation = allocation[:,0]
        self.Put_allocation = allocation[:,-1]

        self.loss_threshold = 1000
        self.vec_final_pnl = np.vectorize(self.final_pnl)
        self.e_pnl = self.expected_pnl() 
#        self.e_utility = self.expected_utility()
        self.pnl_space = self.pnl_space()
        self.Max_Profit = max(self.pnl_space)
        self.Max_Loss = -1*(min(self.pnl_space))
        self.sigma = self.Option_Chain.sigma
        self.name = name
        self.Prob_profit = self.prob_profit()
        self.Prob_loss = self.prob_loss()
        
        Greeks = ["Delta","Gamma","Theta","Vega","Rho"]
        for greek in Greeks:
            call_att = getattr(self.Option_Chain, "Call_"+greek)
            put_att = getattr(self.Option_Chain, "Put_"+greek)
            call_c = self.Option_Chain.lot_size*(np.sum(self.Call_allocation*call_att))
            put_c = self.Option_Chain.lot_size*(np.sum(self.Put_allocation*put_att))
            setattr(self,"Strategy_"+greek, call_c+put_c)
        
#        self.plot_pnl()

       
    def payoff(self,S_T):
        call_payoff = self.Option_Chain.lot_size*(np.sum(self.Call_allocation*np.maximum((S_T-self.Option_Chain.Call_Strike),0)))
        put_payoff = self.Option_Chain.lot_size*(np.sum(self.Put_allocation*np.maximum((self.Option_Chain.Put_Strike-S_T),0)))
        final_payoff = call_payoff+put_payoff
        return final_payoff
    
    def initial_cost(self):

        call_cost = self.Option_Chain.Call_Ask*((self.Call_allocation>0).astype(int))+self.Option_Chain.Call_Bid*((self.Call_allocation<=0).astype(int))
        put_cost = self.Option_Chain.Put_Ask*((self.Put_allocation>0).astype(int))+self.Option_Chain.Put_Bid*((self.Put_allocation<=0).astype(int))
        
        total_call_cost = np.sum(self.Option_Chain.lot_size*(self.Call_allocation*call_cost))
        total_put_cost = np.sum(self.Option_Chain.lot_size*(self.Put_allocation*put_cost))
        return total_call_cost+total_put_cost

    def final_pnl(self,S_T):
        return self.payoff(S_T)-self.initial_cost()
    
    def plot_pnl(self):
        S = np.linspace(self.Option_Chain.Stock_Last - 4*self.sigma, self.Option_Chain.Stock_Last + 4*self.sigma, 1000)   
        pnl = self.vec_final_pnl(S)
        max_loss = round(min(pnl),2)
        e_pnl = self.expected_pnl()
        fig,ax = plt.subplots(1, 1, figsize=(9, 5))
        plt.plot(S,pnl,lw=2.5, color='blue', label = "Final PnL as a function of $S_T$")
        plt.axhline(y=0, color="black", lw = 1)
        plt.axhline(y=max_loss, color="red",linestyle='--', lw = 1, label = "Lowest Pnl = {}".format(max_loss))
        plt.axhline(y = e_pnl  , color = "magenta", linestyle='--', lw =1, label = "$E(Profit) : {}$".format(e_pnl) )
        plt.axvline(x = self.Option_Chain.Stock_Last, color="green", linestyle='--', label = "$S_0 : {}$".format(self.Option_Chain.Stock_Last))
        fmt = '${x:,.2f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)
        ax.xaxis.set_major_formatter(tick)
        plt.xlabel('$S_T$')
        plt.ylabel('Final P & L ')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-')
        ax.grid(which='minor', linestyle=':')
        plt.legend()
        plt.grid(True)
        
        ax2 = ax.twinx()
        color = 'tab:blue'
        mu = self.Option_Chain.Stock_Last
        sigma = self.sigma
        s = self.Option_Chain.lognormal_s
        scale = self.Option_Chain.lognormal_scale
        ax2.set_ylabel('Pdf', color=color)  # we already handled the x-label with ax1
        ax2.vlines(x = mu+sigma, ymin=0, ymax = stats.lognorm.pdf(mu+sigma,s,0,scale), color=color, linestyle='--', label = "$\pm \sigma $")
        ax2.vlines(x = mu-sigma,  ymin=0, ymax = stats.lognorm.pdf(mu-sigma,s,0,scale),color=color, linestyle='--', label = "$\pm \sigma $")  
        ax2.vlines(x = mu+2*sigma,  ymin=0, ymax = stats.lognorm.pdf(mu+2*sigma,s,0,scale),color=color, linestyle='--', label = "$\pm 2\sigma $")
        ax2.vlines(x = mu-2*sigma, ymin=0,  ymax = stats.lognorm.pdf(mu-2*sigma,s,0,scale),color=color, linestyle='--', label = "$\pm 2\sigma $")
        ax2.vlines(x = mu+3*sigma,  ymin=0, ymax = stats.lognorm.pdf(mu+3*sigma,s,0,scale),color=color, linestyle='--', label = "$\pm 3\sigma $")
        ax2.vlines(x = mu-3*sigma,  ymin=0, ymax = stats.lognorm.pdf(mu-3*sigma,s,0,scale),color=color, linestyle='--', label = "$\pm 3\sigma $")
        ax2.plot(S, stats.lognorm.pdf(S, s, 0, scale), color=color, linestyle='--', lw = 2, label = "PDF of $S_T$")
        ax2.tick_params(axis='y', labelcolor=color)
        self.figure = fig
        
        
    def e_curve(self):
        s_paths = self.Option_Chain.S_space
        s_paths_density = self.Option_Chain.S_density
        final_pnl = self.vec_final_pnl(s_paths)
        curve = s_paths_density*final_pnl
        return curve
        
    def utility_curve(self):
        s_paths = self.Option_Chain.S_space
        s_paths_density = self.Option_Chain.S_density
        final_pnl = self.vec_final_pnl(s_paths)
        curve = s_paths_density*np.exp(final_pnl)
        return curve
        
    def expected_utility(self):
        curve = self.utility_curve()
        expected_util = (self.Option_Chain.S_space[1]-self.Option_Chain.S_space[0])*(sum(curve)-0.5*curve[0]-0.5*curve[-1])
        return round(expected_util,2)
    
    def expected_pnl(self):
        curve = self.e_curve()
        h = (self.Option_Chain.S_space[1]-self.Option_Chain.S_space[0])
        expected_pnl = h*(sum(curve)-0.5*curve[0]-0.5*curve[-1])
        return round(expected_pnl,2)
    
    def pnl_st(self):
        return self.final_pnl(self.Option_Chain.Stock_Last)
    
    def pnl_space(self):
        return self.vec_final_pnl(self.Option_Chain.S_space)
    
    def prob_profit(self):
        S = self.Option_Chain.S_space
        S_dens = self.Option_Chain.S_density
        pnl_curve = self.pnl_space.copy()
        S_dens_pos = S_dens.copy()
        S_dens_pos[pnl_curve<=0] = 0 
        prob_profit = (S[1]-S[0])*(sum(S_dens_pos)- 0.5*S_dens_pos[0]-0.5*S_dens_pos[-1])
        return round(100*prob_profit,2)
    
    def prob_loss(self):
        S = self.Option_Chain.S_space
        S_dens = self.Option_Chain.S_density
        pnl_curve = self.pnl_space.copy()
        S_dens_neg = S_dens.copy()
        S_dens_neg[pnl_curve>=0] = 0 
        prob_loss = (S[1]-S[0])*(sum(S_dens_neg)- 0.5*S_dens_neg[0]-0.5*S_dens_neg[-1])
        return round(100*prob_loss, 2)
    
    def summary(self):
        strat_summary = dict()
        
        strat_summary["Underlying"] = self.Option_Chain.Name
        strat_summary["Name"] = self.name
        


        strat_summary["Expected PnL"] = self.e_pnl
#        strat_summary["Expected Utility"] = self.e_utility
        
        strat_summary["Cost of Strategy"] = self.initial_cost()
        
        strat_summary["Max_Profit"] = self.Max_Profit
        strat_summary["Max_Loss"] = self.Max_Loss
        
        strat_summary["Prob of Profit"] = self.Prob_profit
        strat_summary["Prob of Loss"] = self.Prob_loss
        
        strat_summary["Exp_Pnl/Max_Loss"]=self.e_pnl/self.Max_Loss
        
        Greeks = ["Delta","Gamma","Theta","Vega","Rho"]
        
        for greek in Greeks:
            strat_summary["Strategy_"+greek] = getattr(self,"Strategy_"+greek)
            
            

        
        
        Call_strikes = self.Option_Chain.Call_Strike
        Put_strikes = self.Option_Chain.Put_Strike
        
        for i in range(len(Call_strikes)):
            strat_summary["C_"+str(Call_strikes[i])] = self.Call_allocation[i]
        for i in range(len(Put_strikes)):
            strat_summary["P_"+str(Put_strikes[i])] = self.Put_allocation[i]
        
        
        return strat_summary
    

        
        
        
        
    
"""
#######################################################################################
                                    Calculations                 
#######################################################################################
""" 

        
        
        
Assets = list(final_dataset['Group'].unique())


All_Option_Chains = list()
print("List of all Underlyings")

for i in range(len(Assets)): 
    print(i+1, Assets[i])
    All_Option_Chains.append(Option_chain(Assets[i],final_dataset))


tic = datetime.now()


All_Strategies = list()
All_Strategies_Summary = list()

tic = datetime.now()

#for chain in All_Option_Chains:
for i in range(len(All_Option_Chains)):
    chain = All_Option_Chains[i]
    print("\n Processing ",i+1,"/",len(All_Option_Chains), "Underlying", chain.Name)

    Master_List_Strategies = list()
    Master_List_Strategy_Summary = pd.DataFrame()
    #chain = Option_chain(Assets[2],final_dataset)

    
    if "Bull Call Spread" in Strategies:
        Strategy_name ="Bull Call Spread"
        print("\t Processing ", Strategy_name, " Strategy")
        call_1_pos = list(np.arange(chain.Call_total))
        call_2_pos = list(np.arange(chain.Call_total))
        call_1_quantity = list(np.arange(1,max_quantity_per_leg+1))
        call_2_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        iterables = [call_1_pos,call_2_pos,call_1_quantity,call_2_quantity]
        
        bull_call_spread_strat = list()
        bull_call_spread = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, quan_1, quan_2 = t
            if pos_1<pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,0] = quan_2
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    bull_call_spread_strat.append(strat)
                    bull_call_spread.append(details)
                
        if len(bull_call_spread)>0:
            bull_call_spread_df = pd.DataFrame(bull_call_spread)
            bull_call_spread_df = bull_call_spread_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bull_call_spread_df)
            Master_List_Strategies.append(bull_call_spread_strat)  
            print("\t \t Added ", len(bull_call_spread), " Strategies")
    
    
    
    if "Bear Call Spread" in Strategies:
        Strategy_name = "Bear Call Spread"
        print("\t Processing ", Strategy_name, " Strategy")
        call_1_pos = list(np.arange(chain.Call_total))
        call_2_pos = list(np.arange(chain.Call_total))
        call_1_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        call_2_quantity = list(np.arange(1,max_quantity_per_leg+1))
        
        iterables = [call_1_pos,call_2_pos,call_1_quantity,call_2_quantity]
        
        bear_call_spread_strat = list()
        bear_call_spread = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, quan_1, quan_2 = t
            if pos_1<pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,0] = quan_2
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    bear_call_spread_strat.append(strat)
                    bear_call_spread.append(details)
                
        if len(bear_call_spread)>0:
            bear_call_spread_df = pd.DataFrame(bear_call_spread)
            bear_call_spread_df = bear_call_spread_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bear_call_spread_df)
            Master_List_Strategies.append(bear_call_spread_strat)  
            print("\t \t Added ", len(bear_call_spread), " Strategies")
        
    if "Bull Put Spread" in Strategies:
        Strategy_name ="Bull Put Spread"
        print("\t Processing ", Strategy_name, " Strategy")
        put_1_pos = list(np.arange(chain.Call_total))
        put_2_pos = list(np.arange(chain.Call_total))
        put_1_quantity = list(np.arange(1,max_quantity_per_leg+1))
        put_2_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        iterables = [put_1_pos,put_2_pos,put_1_quantity,put_2_quantity]
        
        bull_put_spread_strat = list()
        bull_put_spread = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, quan_1, quan_2 = t
            if pos_1<pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quan_1
                allocation[pos_2,1] = quan_2
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl :
                    bull_put_spread_strat.append(strat)
                    bull_put_spread.append(details)
                
        if len(bull_put_spread)>0:
            bull_put_spread_df = pd.DataFrame(bull_put_spread)
            bull_put_spread_df = bull_put_spread_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bull_put_spread_df)
            Master_List_Strategies.append(bull_put_spread_strat)
            print("\t \t Added ", len(bull_put_spread), " Strategies")
    
    
    if "Bear Put Spread" in Strategies:
        Strategy_name ="Bear Put Spread"
        print("\t Processing ", Strategy_name, " Strategy")
        put_1_pos = list(np.arange(chain.Call_total))
        put_2_pos = list(np.arange(chain.Call_total))
        put_1_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        put_2_quantity = list(np.arange(1,max_quantity_per_leg+1))
        iterables = [put_1_pos,put_2_pos,put_1_quantity,put_2_quantity]
        
        bear_put_spread_strat = list()
        bear_put_spread = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, quan_1, quan_2 = t
            if pos_1<pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quan_1
                allocation[pos_2,1] = quan_2
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl :
                    bear_put_spread_strat.append(strat)
                    bear_put_spread.append(details)
                
        if len(bear_put_spread)>0:       
            bear_put_spread_df = pd.DataFrame(bear_put_spread)
            bear_put_spread_df = bear_put_spread_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bear_put_spread_df)
            Master_List_Strategies.append(bear_put_spread_strat)
            print("\t \t Added ", len(bear_put_spread), " Strategies")
    
          
    
    if "Bull Put Ladder" in Strategies:
        Strategy_name ="Bull Put Ladder"
        print("\t Processing ", Strategy_name, " Strategy")
        put_1_pos = list(np.arange(chain.Call_total))
        put_2_pos = list(np.arange(chain.Call_total))
        put_3_pos = list(np.arange(chain.Call_total))
        put_1_quantity = list(np.arange(1,max_quantity_per_leg+1))
        put_2_quantity = list(np.arange(1,max_quantity_per_leg+1))
        put_3_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        iterables = [put_1_pos,put_2_pos, put_3_pos, put_1_quantity,put_2_quantity, put_3_quantity]
        
        bull_put_ladder_strat = list()
        bull_put_ladder = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2, quan_3 = t
            if pos_1<pos_2 and pos_2 < pos_3 :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quan_1
                allocation[pos_2,1] = quan_2
                allocation[pos_3,1] = quan_3
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    bull_put_ladder_strat.append(strat)
                    bull_put_ladder.append(details)
                
        if len(bull_put_ladder)>0:
            bull_put_ladder_df = pd.DataFrame(bull_put_ladder)
            bull_put_ladder_df = bull_put_ladder_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bull_put_ladder_df)
            Master_List_Strategies.append(bull_put_ladder_strat)
            print("\t \t Added ", len(bull_put_ladder), " Strategies")
    
    
    if "Bear Call Ladder" in Strategies:
        Strategy_name ="Bear Call Ladder"
        print("\t Processing ", Strategy_name, " Strategy")
        call_1_pos = list(np.arange(chain.Call_total))
        call_2_pos = list(np.arange(chain.Call_total))
        call_3_pos = list(np.arange(chain.Call_total))
        call_1_quantity = list(-1*np.arange(1,max_quantity_per_leg+1))
        call_2_quantity = list(np.arange(1,max_quantity_per_leg+1))
        call_3_quantity = list(np.arange(1,max_quantity_per_leg+1))
        iterables = [call_1_pos,call_2_pos, call_3_pos, call_1_quantity,call_2_quantity, call_3_quantity]
        
        bear_call_ladder_strat = list()
        bear_call_ladder = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2, quan_3 = t
            if pos_1<pos_2 and pos_2 < pos_3 :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,0] = quan_2
                allocation[pos_3,0] = quan_3
                strat = Strategy(allocation,chain,Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl :
                    bear_call_ladder_strat.append(strat)
                    bear_call_ladder.append(details)
                
        if len(bear_call_ladder)>0:        
            bear_call_ladder_df = pd.DataFrame(bear_call_ladder)
            bear_call_ladder_df = bear_call_ladder_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(bear_call_ladder_df)
            Master_List_Strategies.append(bear_call_ladder_strat)  
            print("\t \t Added ", len(bear_call_ladder), " Strategies")
    
    if "Long Straddle" in Strategies:
        Strategy_name = "Long Straddle"
        print("\t Processing ", Strategy_name, " Strategy")
        pos = list(np.arange(chain.Call_total))
        call_quan = list(np.arange(1,max_quantity_per_leg+1))
        put_quan = list(np.arange(1,max_quantity_per_leg+1))
        
        iterables = [pos,call_quan, put_quan]
        
        long_straddle_strat = list()
        long_straddle = list()
        for t in itertools.product(*iterables):
            pos,quan_1, quan_2 = t
            allocation = np.zeros((chain.Call_total,2))
            allocation[pos,0] = quan_1
            allocation[pos,1] = quan_2
            strat = Strategy(allocation, chain, Strategy_name)
            details = strat.summary()
            if details["Expected PnL"] > min_e_pnl:
                long_straddle_strat.append(strat)
                long_straddle.append(details)
        if len(long_straddle)>0:
            long_straddle_df = pd.DataFrame(long_straddle)
            long_straddle_df = long_straddle_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_straddle_df)
            Master_List_Strategies.append(long_straddle_strat)   
            print("\t \t Added ", len(long_straddle), " Strategies")
    
    
    if "Long Strangle" in Strategies:
        Strategy_name = "Long Strangle"
        print("\t Processing ", Strategy_name, " Strategy")
        call_pos = list(np.arange(chain.Call_total))
        put_pos = list(np.arange(chain.Call_total))
        call_quan = list(np.arange(1,max_quantity_per_leg+1))
        put_quan = list(np.arange(1,max_quantity_per_leg+1))
        
        iterables = [call_pos,put_pos,call_quan, put_quan]
        
        long_strangle_strat = list()
        long_strangle = list()
        for t in itertools.product(*iterables):
            pos_1,pos_2,quan_1, quan_2 = t
            if pos_1 < pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,1] = quan_2
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_strangle_strat.append(strat)
                    long_strangle.append(details)
        if len(long_strangle)>0:
            long_strangle_df = pd.DataFrame(long_strangle)
            long_strangle_df = long_strangle_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_strangle_df)
            Master_List_Strategies.append(long_strangle_strat)
            print("\t \t Added ", len(long_strangle), " Strategies")
    
    if "Short Straddle" in Strategies:
        Strategy_name = "Short Straddle"
        print("\t Processing ", Strategy_name, " Strategy")
        pos = list(np.arange(chain.Call_total))
        call_quan = list(-1*np.arange(1,max_quantity_per_leg+1))
        put_quan = list(-1*np.arange(1,max_quantity_per_leg+1))
        
        iterables = [pos, call_quan, put_quan]
        
        short_straddle_strat = list()
        short_straddle = list()
        for t in itertools.product(*iterables):
            pos,quan_1, quan_2 = t
            allocation = np.zeros((chain.Call_total,2))
            allocation[pos,0] = quan_1
            allocation[pos,1] = quan_2
            strat = Strategy(allocation, chain, Strategy_name)
            details = strat.summary()
            if details["Expected PnL"] > min_e_pnl:
                short_straddle_strat.append(strat)
                short_straddle.append(details)
        if len(short_straddle)>0:
            short_straddle_df = pd.DataFrame(short_straddle)
            short_straddle_df = short_straddle_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_straddle_df)
            Master_List_Strategies.append(short_straddle_strat)
            print("\t \t Added ", len(short_straddle), " Strategies")
    
    
    
    if "Short Strangle" in Strategies:
        Strategy_name = "Short Strangle"
        print("\t Processing ", Strategy_name, " Strategy")
        call_pos = list(np.arange(chain.Call_total))
        put_pos = list(np.arange(chain.Call_total))
        call_quan = list(-1*np.arange(1,max_quantity_per_leg+1))
        put_quan = list(-1*np.arange(1,max_quantity_per_leg+1))
        
        iterables = [call_pos,put_pos,call_quan, put_quan]
        
        short_strangle_strat = list()
        short_strangle = list()
        for t in itertools.product(*iterables):
            pos_1,pos_2,quan_1, quan_2 = t
            if pos_1 < pos_2:
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,1] = quan_2
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_strangle_strat.append(strat)
                    short_strangle.append(details)
        
        if len(short_strangle)>0:
            short_strangle_df = pd.DataFrame(short_strangle)
            short_strangle_df = short_strangle_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_strangle_df)
            Master_List_Strategies.append(short_strangle_strat)
            print("\t \t Added ", len(short_strangle), " Strategies")
    
    
    if "Long Call Butterfly" in Strategies:
        Strategy_name = "Long Call Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        call_pos_1 = list(np.arange(chain.Call_total))
        call_pos_2 = list(np.arange(chain.Call_total))
        call_pos_3 = list(np.arange(chain.Call_total))
        call_quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        call_quan_2 = list(-1*np.arange(2,max_quantity_per_leg+1))
    
        
        iterables = [call_pos_1, call_pos_2, call_pos_3, call_quan_1, call_quan_2]
        
        long_call_butterfly_strat = list()
        long_call_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and quan_1 < abs(quan_2) :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,0] = quan_2
                allocation[pos_3,0] = quan_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_call_butterfly_strat.append(strat)
                    long_call_butterfly.append(details)
        if len(long_call_butterfly)>0:
            long_call_butterfly_df = pd.DataFrame(long_call_butterfly)
            long_call_butterfly_df = long_call_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_call_butterfly_df)
            Master_List_Strategies.append(long_call_butterfly_strat)  
            print("\t \t Added ", len(long_call_butterfly), " Strategies")
    
    
    if "Long Put Butterfly" in Strategies:
        Strategy_name = "Long Put Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        put_pos_1 = list(np.arange(chain.Call_total))
        put_pos_2 = list(np.arange(chain.Call_total))
        put_pos_3 = list(np.arange(chain.Call_total))
        put_quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        put_quan_2 = list(-1*np.arange(2,max_quantity_per_leg+1))
    
        
        iterables = [put_pos_1, put_pos_2, put_pos_3, put_quan_1, put_quan_2]
        
        long_put_butterfly_strat = list()
        long_put_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and quan_1 < abs(quan_2) :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quan_1
                allocation[pos_2,1] = quan_2
                allocation[pos_3,1] = quan_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_put_butterfly_strat.append(strat)
                    long_put_butterfly.append(details)
        if len(long_put_butterfly)>0:
            long_put_butterfly_df = pd.DataFrame(long_put_butterfly)
            long_put_butterfly_df = long_put_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_put_butterfly_df)
            Master_List_Strategies.append(long_put_butterfly_strat)
            print("\t \t Added ", len(long_put_butterfly), " Strategies")
    
    
    
    
    
    if "Short Call Butterfly" in Strategies:
        Strategy_name = "Short Call Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        call_pos_1 = list(np.arange(chain.Call_total))
        call_pos_2 = list(np.arange(chain.Call_total))
        call_pos_3 = list(np.arange(chain.Call_total))
        call_quan_1 = list(-1*np.arange(1,max_quantity_per_leg+1))
        call_quan_2 = list(np.arange(2,max_quantity_per_leg+1))
    
        
        iterables = [call_pos_1, call_pos_2, call_pos_3, call_quan_1, call_quan_2]
        
        short_call_butterfly_strat = list()
        short_call_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and abs(quan_1) < quan_2 :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = quan_1
                allocation[pos_2,0] = quan_2
                allocation[pos_3,0] = quan_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_call_butterfly_strat.append(strat)
                    short_call_butterfly.append(details)
        if len(short_call_butterfly)>0:
            short_call_butterfly_df = pd.DataFrame(short_call_butterfly)
            short_call_butterfly_df = short_call_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_call_butterfly_df)
            Master_List_Strategies.append(short_call_butterfly_strat)
            print("\t \t Added ", len(short_call_butterfly), " Strategies")
    
    
    
    
    if "Short Put Butterfly" in Strategies:
        Strategy_name = "Short Put Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        put_pos_1 = list(np.arange(chain.Call_total))
        put_pos_2 = list(np.arange(chain.Call_total))
        put_pos_3 = list(np.arange(chain.Call_total))
        put_quan_1 = list(-1*np.arange(1,max_quantity_per_leg+1))
        put_quan_2 = list(np.arange(2,max_quantity_per_leg+1))
    
        
        iterables = [put_pos_1, put_pos_2, put_pos_3, put_quan_1, put_quan_2]
        
        short_put_butterfly_strat = list()
        short_put_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quan_1, quan_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and abs(quan_1) < quan_2 :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quan_1
                allocation[pos_2,1] = quan_2
                allocation[pos_3,1] = quan_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_put_butterfly_strat.append(strat)
                    short_put_butterfly.append(details)
        if len(short_put_butterfly)>0:
            short_put_butterfly_df = pd.DataFrame(short_put_butterfly)
            short_put_butterfly_df = short_put_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_put_butterfly_df)
            Master_List_Strategies.append(short_put_butterfly_strat)
            print("\t \t Added ", len(short_put_butterfly), " Strategies")
    
    
    
    if "Long Iron Butterfly" in Strategies:
        Strategy_name = "Long Iron Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, quan_1, quan_2]
        
        long_iron_butterfly_strat = list()
        long_iron_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quant_1, quant_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quant_1
                allocation[pos_2,1] = -1*quant_2
                allocation[pos_2,0] = -1*quant_2
                allocation[pos_3,0] = quant_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_iron_butterfly_strat.append(strat)
                    long_iron_butterfly.append(details)
        if len(long_iron_butterfly)>0:
            long_iron_butterfly_df = pd.DataFrame(long_iron_butterfly)
            long_iron_butterfly_df = long_iron_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_iron_butterfly_df)
            Master_List_Strategies.append(long_iron_butterfly_strat) 
            print("\t \t Added ", len(long_iron_butterfly), " Strategies")
    
    
    if "Short Iron Butterfly" in Strategies:
        Strategy_name = "Short Iron Butterfly"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, quan_1, quan_2]
        
        short_iron_butterfly_strat = list()
        short_iron_butterfly = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, quant_1 , quant_2 = t
            if pos_1 < pos_2 and pos_2 < pos_3  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = -1*quant_1
                allocation[pos_2,1] = quant_2
                allocation[pos_2,0] = quant_2
                allocation[pos_3,0] = -1*quant_1
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_iron_butterfly_strat.append(strat)
                    short_iron_butterfly.append(details)
        if len(short_iron_butterfly)>0:
            short_iron_butterfly_df = pd.DataFrame(short_iron_butterfly)
            short_iron_butterfly_df = short_iron_butterfly_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_iron_butterfly_df)
            Master_List_Strategies.append(short_iron_butterfly_strat)
            print("\t \t Added ", len(short_iron_butterfly), " Strategies")
    
    
    
    
    
    
    
    if "Long Call Condor" in Strategies:
        Strategy_name = "Long Call Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        long_call_condor_strat = list()
        long_call_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, q_1, q_2, q_3, q_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = q_1
                allocation[pos_2,0] = -1*q_2
                allocation[pos_3,0] = -1*q_3
                allocation[pos_4,0] = q_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_call_condor_strat.append(strat)
                    long_call_condor.append(details)
        if len(long_call_condor)>0:
            long_call_condor_df = pd.DataFrame(long_call_condor)
            long_call_condor_df = long_call_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_call_condor_df)
            Master_List_Strategies.append(long_call_condor_strat)  
            print("\t \t Added ", len(long_call_condor), " Strategies")
    
    
    if "Long Put Condor" in Strategies:
        Strategy_name = "Long Put Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        long_put_condor_strat = list()
        long_put_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, q_1, q_2, q_3, q_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = q_1
                allocation[pos_2,1] = -1*q_2
                allocation[pos_3,1] = -1*q_3
                allocation[pos_4,1] = q_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_put_condor_strat.append(strat)
                    long_put_condor.append(details)
        if len(long_put_condor)>0:
            long_put_condor_df = pd.DataFrame(long_put_condor)
            long_put_condor_df = long_put_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_put_condor_df)
            Master_List_Strategies.append(long_put_condor_strat)  
            print("\t \t Added ", len(long_put_condor), " Strategies")
    
    
    
    if "Short Call Condor" in Strategies:
        Strategy_name = "Short Call Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        short_call_condor_strat = list()
        short_call_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, q_1, q_2, q_3, q_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = -1*q_1
                allocation[pos_2,0] = q_2
                allocation[pos_3,0] = q_3
                allocation[pos_4,0] = -1*q_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_call_condor_strat.append(strat)
                    short_call_condor.append(details)
        if len(short_call_condor)>0:
            short_call_condor_df = pd.DataFrame(short_call_condor)
            short_call_condor_df = long_call_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_call_condor_df)
            Master_List_Strategies.append(short_call_condor_strat)
            print("\t \t Added ", len(short_call_condor), " Strategies")
    
    
    if "Short Put Condor" in Strategies:
        Strategy_name = "Short Put Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        short_put_condor_strat = list()
        short_put_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, q_1, q_2, q_3, q_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = -1*q_1
                allocation[pos_2,1] = q_2
                allocation[pos_3,1] = q_3
                allocation[pos_4,1] = -1*q_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_put_condor_strat.append(strat)
                    short_put_condor.append(details)
        if len(short_put_condor)>0:
            short_put_condor_df = pd.DataFrame(short_put_condor)
            short_put_condor_df = short_put_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_put_condor_df)
            Master_List_Strategies.append(short_put_condor_strat)
            print("\t \t Added ", len(short_put_condor), " Strategies")
    
    
    
    
    if "Long Iron Condor" in Strategies:
        Strategy_name = "Long Iron Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        long_iron_condor_strat = list()
        long_iron_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, quant_1, quant_2, quant_3, quant_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = quant_1
                allocation[pos_2,1] = -1*quant_2
                allocation[pos_3,0] = -1*quant_3
                allocation[pos_4,0] = quant_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_iron_condor_strat.append(strat)
                    long_iron_condor.append(details)
        if len(long_iron_condor)>0:
            long_iron_condor_df = pd.DataFrame(long_iron_condor)
            long_iron_condor_df = long_iron_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_iron_condor_df)
            Master_List_Strategies.append(long_iron_condor_strat)
            print("\t \t Added ", len(long_iron_condor), " Strategies")
            
    if "Short Iron Condor" in Strategies:
        Strategy_name = "Short Iron Condor"
        print("\t Processing ", Strategy_name, " Strategy")
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        strike_pos_3 = list(np.arange(chain.Call_total))
        strike_pos_4 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
    
        
        iterables = [strike_pos_1, strike_pos_2, strike_pos_3, strike_pos_4, quan_1, quan_2, quan_3, quan_4]
        
        short_iron_condor_strat = list()
        short_iron_condor = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, pos_3, pos_4, quant_1, quant_2, quant_3, quant_4 = t
            if pos_1 < pos_2 and pos_2 < pos_3 and pos_3 < pos_4  :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,1] = -1*quant_1
                allocation[pos_2,1] =  quant_2
                allocation[pos_3,0] =  quant_3
                allocation[pos_4,0] = -1*quant_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    short_iron_condor_strat.append(strat)
                    short_iron_condor.append(details)
        if len(short_iron_condor)>0:
            short_iron_condor_df = pd.DataFrame(short_iron_condor)
            short_iron_condor_df = short_iron_condor_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(short_iron_condor_df)
            Master_List_Strategies.append(short_iron_condor_strat)
            print("\t \t Added ", len(short_iron_condor), " Strategies")
    
    if "Long Box" in Strategies:
        Strategy_name = "Long Box"
        print("\t Processing ", Strategy_name, " Strategy")
        
        strike_pos_1 = list(np.arange(chain.Call_total))
        strike_pos_2 = list(np.arange(chain.Call_total))
        quan_1 = list(np.arange(1,max_quantity_per_leg+1))
        quan_2 = list(np.arange(1,max_quantity_per_leg+1))
        quan_3 = list(np.arange(1,max_quantity_per_leg+1))
        quan_4 = list(np.arange(1,max_quantity_per_leg+1))
        
        iterables = [strike_pos_1, strike_pos_2, quan_1, quan_2, quan_3, quan_4]
        
        long_box_strat = list()
        long_box = list()
        for t in itertools.product(*iterables):
            pos_1, pos_2, q_1, q_2, q_3, q_4= t
            if pos_1 < pos_2   :
                allocation = np.zeros((chain.Call_total,2))
                allocation[pos_1,0] = q_1
                allocation[pos_1,1] = -1*q_2
                allocation[pos_2,1] = q_3
                allocation[pos_2,0] = -1*q_4
                strat = Strategy(allocation, chain, Strategy_name)
                details = strat.summary()
                if details["Expected PnL"] > min_e_pnl:
                    long_box_strat.append(strat)
                    long_box.append(details)
        if len(long_box)>0:
            long_box_df = pd.DataFrame(long_box)
            long_box_df = long_box_df.sort_values(by=["Exp_Pnl/Max_Loss","Max_Profit"], ascending=False)
            Master_List_Strategy_Summary = Master_List_Strategy_Summary.append(long_box_df)
            Master_List_Strategies.append(long_box_strat)  
            print("\t \t Added ", len(long_box), " Strategies")


    """
    Append all strategies of Underlying
    """
    All_Strategies.append(Master_List_Strategies)
    All_Strategies_Summary.append(Master_List_Strategy_Summary)



toc = datetime.now()

print("\n Time Elapsed :", toc-tic)
    
   
now = datetime.now()


path = "./"+now.strftime("%Y_%m_%d_%H_%M_%S")
try:
    os.mkdir(path)
except OSError:
    print ("\n Creation of the directory %s failed" % path)
else:
    print ("\n Successfully created the directory %s " % path)

for i in range(len(Assets)):
    df = All_Strategies_Summary[i]
    outname = Assets[i]+".csv"
    fullname = os.path.join(path, outname)   
    df.to_csv(fullname)















### Check 
#for i in range(5):
#
#allocation = np.zeros((5,2))
#allocation[1,0]= 1
#allocation[4,1] = 2
##allocation[4,0] = 1
#opt_strategy = Strategy(allocation,chain,"New")
#pnl = opt_strategy.final_pnl(chain.Stock_Last)
#payoff = opt_strategy.payoff(chain.Stock_Last)
#cos = -1*(pnl-payoff)
#
#prob = opt_strategy.prob_profit()
#
##total_pnl  = opt_strategy.pnl_space()
#print("PnL at S_T :", pnl)
#print(payoff)
#print(cos)
#print("Expected PnL :", opt_strategy.expected_pnl() )
#opt_strategy.plot_pnl()












