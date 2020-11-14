# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 17:27:08 2020

@author: Pavan
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import random 
import os
from tsmoothie.smoother import *
# last_available_date = "2020-10-03"

# symbol = "^GSPC"

def multi_step_pred(symbol, last_available_date, path,name,plot_status=True):
    
    if symbol=="SPX" or symbol == "SPXW":
        symbol = "^GSPC"
    eod_data = yf.download(symbol, start="2018-06-01", end= last_available_date)
    
    for col in eod_data.columns:
        eod_data[col] = pd.to_numeric(eod_data[col],errors='coerce')
    
    eod_data["Date"] = pd.to_datetime(eod_data.index, format="%Y-%m-%d")
    
    smoother = KalmanSmoother(component='level_trend', 
                          component_noise={'level':0.2, 'trend':0.2})
    # Select only the important features i.e. the date and price
    data = eod_data[["Date","Close"]] # select Date and Price
    # Rename the features: These names are NEEDED for the model fitting
    data['Close'] = np.squeeze(smoother.smooth(data['Close']).smooth_data)
    data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
    
    cycle=361
    from fbprophet import Prophet
    m = Prophet(
        growth="linear",
        # holidays=holidays,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.95
        ).add_seasonality(
            name='monthly',
            period= cycle/12,
            fourier_order=20
        ).add_seasonality(
            name='daily',
            period=1,
            fourier_order=20
        ).add_seasonality(
            name='weekly',
            period=cycle/52,
            fourier_order=20
        ).add_seasonality(
            name='yearly',
            period=cycle,
            fourier_order=20
        ).add_seasonality(
            name='quarterly',
            period=cycle/4,
            fourier_order=10)
            
    m.add_country_holidays(country_name='US')
    # m.add_seasonality('self_define_cycle',period=cycle,fourier_order=8,mode=mode)
    m.fit(data) # fit the model using all dat
    
    future = m.make_future_dataframe(periods=5) #we need to specify the number of days in future
    prediction = m.predict(future)
    final_res = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)
    final_res = final_res.rename(columns = {"ds":"Date"}) #renaming the columns of the dataset
    result = pd.merge(final_res, data, how='left', on=['Date'])
    result['pct_change'] = result['yhat'].pct_change()
    result['log_ret'] = np.log(result['yhat'].astype('float64')/result['yhat'].astype('float64').shift(1))
    
    result_forecast = result[result['y'].isnull()]
    result_forecast['cum_ret'] = result_forecast['log_ret'].cumsum()
    result_forecast.reset_index(drop=True, inplace=True)
    
    if plot_status==True:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        m.plot(prediction)
        plt.title("Prediction of "+symbol+"  using the Prophet")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        # plt.show()
        file_name = name+".png"
        plt.savefig(os.path.join(path, file_name))
        plt.close(fig)
        
        # Python
        # m.plot_components(prediction)
    
    return result, result_forecast

def wwma(values, n):
    """
     J. Welles Wilder's EMA 
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, n=14):
    data = df.copy()
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['ATR'] = wwma(tr, n)
    return df



def multi_step_pred_vol(symbol, last_available_date, path,name,plot_status=True):
    if symbol=="SPX" or symbol == "SPXW":
        symbol = "^GSPC"
    eod_data = yf.download(symbol, start="2018-06-01", end= last_available_date)
    
    smoother = KalmanSmoother(component='level_trend', 
                          component_noise={'level':0.2, 'trend':0.2})
    for col in eod_data.columns:
        eod_data[col] = pd.to_numeric(eod_data[col],errors='coerce')
    
    eod_data["Date"] = pd.to_datetime(eod_data.index, format="%Y-%m-%d")
    eod_data = atr(eod_data)
    
    # Select only the important features i.e. the date and price
    data = eod_data[["Date","ATR"]] # select Date and Price
    # Rename the features: These names are NEEDED for the model fitting
    data['ATR'] = np.squeeze(smoother.smooth(data['ATR']).smooth_data)
    data = data.rename(columns = {"Date":"ds","ATR":"y"}) #renaming the columns of the dataset
    
    cycle=361
    from fbprophet import Prophet
    m = Prophet(
        growth="linear",
        # holidays=holidays,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=0.95
        ).add_seasonality(
            name='monthly',
            period= cycle/12,
            fourier_order=20
        ).add_seasonality(
            name='daily',
            period=1,
            fourier_order=20
        ).add_seasonality(
            name='weekly',
            period=cycle/52,
            fourier_order=20
        ).add_seasonality(
            name='yearly',
            period=cycle,
            fourier_order=20
        ).add_seasonality(
            name='quarterly',
            period=cycle/4,
            fourier_order=10)
            
    m.add_country_holidays(country_name='US')
    # m.add_seasonality('self_define_cycle',period=cycle,fourier_order=8,mode=mode)
    m.fit(data) # fit the model using all dat
    
    future = m.make_future_dataframe(periods=5) #we need to specify the number of days in future
    prediction = m.predict(future)
    final_res = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)
    final_res = final_res.rename(columns = {"ds":"Date"}) #renaming the columns of the dataset
    result = pd.merge(final_res, data, how='left', on=['Date'])
    result['pct_change'] = result['yhat'].pct_change()
    result['log_ret'] = np.log(result['yhat'].astype('float64')/result['yhat'].astype('float64').shift(1))
    
    result_forecast = result[result['y'].isnull()]
    result_forecast['cum_ret'] = result_forecast['log_ret'].cumsum()
    result_forecast.reset_index(drop=True, inplace=True)
    
    if plot_status==True:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        m.plot(prediction)
        plt.title("Prediction of "+symbol+"  ATR(14) using the Prophet")
        plt.xlabel("Date")
        plt.ylabel("ATR(14)")
        # plt.show()
        file_name = name+"_vol_pred.png"
        plt.savefig(os.path.join(path, file_name))
        plt.close(fig)
        
        # Python
        # m.plot_components(prediction)
    
    return result, result_forecast
