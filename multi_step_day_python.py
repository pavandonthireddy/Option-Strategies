# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 23:45:15 2020

@author: Pavan
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import random 

last_available_date = "2020-10-02"

symbol = "^GSPC"
eod_data = yf.download(symbol, start="2016-06-01", end= last_available_date)

for col in eod_data.columns:
    eod_data[col] = pd.to_numeric(eod_data[col],errors='coerce')

eod_data["Date"] = pd.to_datetime(eod_data.index, format="%Y-%m-%d")


# Select only the important features i.e. the date and price
data = eod_data[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset

# cycle=365.25
# mode='additive'
from fbprophet import Prophet
# m = Prophet(
#     growth="linear",
#     # holidays=holidays,
#     seasonality_mode="multiplicative",
#     changepoint_prior_scale=90,
#     seasonality_prior_scale=90,
#     ###cap=3.00,
#     ###floor=.65*125,
#     holidays_prior_scale=90,
#     daily_seasonality=False,
#     weekly_seasonality=False,
#     yearly_seasonality=False,
#     ).add_seasonality(
#         name='monthly',
#         period= cycle/12,
#         fourier_order=12
#     ).add_seasonality(
#         name='daily',
#         period=1,
#         fourier_order=15
#     ).add_seasonality(
#         name='weekly',
#         period=cycle/52,
#         fourier_order=20
#     ).add_seasonality(
#         name='yearly',
#         period=cycle,
#         fourier_order=20
#     ).add_seasonality(
#         name='quarterly',
#         period=cycle/4,
#         fourier_order=5,
#         prior_scale=15)
        
# m.add_country_holidays(country_name='US')
# # m.add_seasonality('self_define_cycle',period=cycle,fourier_order=8,mode=mode)
# m.fit(data) # fit the model using all dat

# future = m.make_future_dataframe(periods=10) #we need to specify the number of days in future
# prediction = m.predict(future)
# final_res = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(20)
# final_res = final_res.rename(columns = {"ds":"Date"}) #renaming the columns of the dataset
# result = pd.merge(final_res, data, how='left', on=['Date'])

# m.plot(prediction)
# plt.title("Prediction of the SPX using the Prophet")
# plt.xlabel("Date")
# plt.ylabel("Close Price")
# plt.show()

# # Python
# m.plot_components(prediction)


from sklearn.model_selection import ParameterGrid
params_grid = {#'seasonality_mode':('multiplicative','additive'),
                # 'changepoint_prior_scale':[0.03,0.04,0.05],
               'holidays_prior_scale':[0.01,0.1,0.5,1,5,10,20,25,30,50,75,100],
               # 'seasonality_prior_scale':[0.01,0.05,0.1,0.5,1,5,10,20,25,30,50,75,100],
               # 'monthly_fourier_order' : [5,10,15,20,25,30,40,50,60,80],
               # 'daily_fourier_order' : [5,10,15,20,25,30,40,50,60,80],
               # 'weekly_fourier_order' : [5,10,15,20,25,30,40],
               # 'yearly_fourier_order' : [5,10,15,20,25,30,40,50,60,80],
               # 'quarterly_fourier_order' : [8,12],
              # 'cycle': [360.5,361,361.5,362,362.5]
              }
grid = ParameterGrid(params_grid)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

strt='2020-05-01'
end='2020-10-01'
tr_data = data[data['ds']<=strt]
model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
for p in grid:
    test = pd.DataFrame()
    print(p)
    random.seed(123)
    train_model = Prophet(
        growth="linear",
        # holidays=holidays,
        seasonality_mode='multiplicative',#p['seasonality_mode'],
        changepoint_prior_scale=0.05,#p['changepoint_prior_scale'],
        seasonality_prior_scale=10,#p['seasonality_prior_scale'],
        ###cap=3.00,
        ###floor=.65*125,
        holidays_prior_scale=10,#p['holidays_prior_scale'],
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        ).add_seasonality(
            name='monthly',
            period= 361/5,#p['cycle']/12,
            fourier_order=20,#p['monthly_fourier_order']
        ).add_seasonality(
            name='daily',
            period=1,
            fourier_order=20,#p['daily_fourier_order']
        ).add_seasonality(
            name='weekly',
            period=361/52,#p['cycle']/52,
            fourier_order=20,#p['weekly_fourier_order']
        ).add_seasonality(
            name='yearly',
            period=361,#p['cycle'],
            fourier_order=20,#p['yearly_fourier_order']
        ).add_seasonality(
            name='quarterly',
            period=361/4,#p['cycle']/4,
            fourier_order=10)#p['quarterly_fourier_order'])
    train_model.add_country_holidays(country_name='US')
    train_model.fit(tr_data)
    train_forecast = train_model.make_future_dataframe(periods=106, freq='D',include_history = False)
    train_forecast = train_model.predict(train_forecast)
    test=train_forecast[['ds','yhat']]
    Actual = data[(data['ds']>strt) & (data['ds']<=end)]
    MAPE = mean_absolute_percentage_error(Actual['y'],abs(test['yhat']))
    print('Mean Absolute Percentage Error(MAPE)------------------------------------',MAPE)
    model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
