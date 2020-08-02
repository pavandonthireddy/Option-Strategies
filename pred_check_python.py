# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:24:12 2020

@author: Pavan
"""

from prediction_utilities import rstring
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd
pandas2ri.activate()
import matplotlib.pyplot as plt
import os 
import numpy as np

symbol = "AIG"

rfunc=robjects.r(rstring)
r_df=rfunc(symbol)
forecast_df=pandas2ri.ri2py(r_df)


dates = forecast_df["Timestamp"]
dateSelect = pd.to_datetime("'1970-01-01'".replace("'",""))

forecast_df["Timestamp"]= dateSelect + pd.to_timedelta(dates,'d')

forecast_df = forecast_df.set_index('Timestamp')

plot_actual = forecast_df['Price'][forecast_df['Type']=='Actual']

plot_predicted = forecast_df['Price'][forecast_df['Type']=='Predicted']

def plot_actual_pred(actual, predicted, path, name):
    # Plot the predictions for validation set
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('T')
    plt.ylabel('Price')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')
    plt.legend()
    plt.grid(True)
    file_name = name+".png"
    plt.savefig(os.path.join(path, file_name))
    plt.close(fig)
    

#plot_actual_pred(plot_actual.iloc[-50:], plot_predicted, symbol)


forecast_df['log_ret'] = np.log(forecast_df.Price) - np.log(forecast_df.Price.shift(1))

forecast_df['mean'] = forecast_df['log_ret'].rolling(10).mean()
forecast_df['vol'] = forecast_df['log_ret'].rolling(10).std()
forecast_df['z_score'] = (forecast_df['log_ret']-forecast_df['mean'])/forecast_df['vol']
forecast_df['z_score_adj'] = (forecast_df['log_ret'])/forecast_df['vol']

pred_df = forecast_df[forecast_df['Type']=='Predicted']
pred_df['cum_ret'] = pred_df['log_ret'].cumsum()

res = dict()

res['Name'] = symbol
two_week_ret = pred_df.iloc[1,-1]
res['Pred_ret'] = two_week_ret

if two_week_ret>0:
    if two_week_ret <=0.08:
        res['direction'] ='Slight Bullish'
    else:
        res['direction']='Bullish'
else:
    if two_week_ret >= -0.08:
        res['direction'] ='Slight Bearish'
    else:
        res['direction']='Bearish'





    

 

