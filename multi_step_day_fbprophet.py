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

last_available_date = "2020-10-03"

symbol = "^GSPC"
# symbol = "TLT"
eod_data = yf.download(symbol, start="2018-06-01", end= last_available_date)

for col in eod_data.columns:
    eod_data[col] = pd.to_numeric(eod_data[col],errors='coerce')

eod_data["Date"] = pd.to_datetime(eod_data.index, format="%Y-%m-%d")


# Select only the important features i.e. the date and price
data = eod_data[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
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
    interval_width=0.99
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
rows = len(result_forecast)
for i  in range(rows):
    a = round(result_forecast.loc[i,'log_ret']*100,2)
    print("Predicted Return for ", i+1, " day :", a," %")

for i  in range(rows):
    a = round(result_forecast.loc[i,'cum_ret']*100,2)
    b = round(result_forecast.loc[i,'yhat_lower'],2)
    c = round(result_forecast.loc[i,'yhat_upper'],2)
    print("Predicted Cum Return for ", i+1, " day :", a, "%. Bounds are(",b, ", ", c , ')')



m.plot(prediction)
plt.title("Prediction of the SPX using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# Python
m.plot_components(prediction)