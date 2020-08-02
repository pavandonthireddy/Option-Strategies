# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 23:19:37 2020

@author: Pavan
"""



import numpy as np
import datetime as dt 
import pandas as pd
import pandas_datareader.data as web
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from datetime import datetime

now = datetime.now()
path = "./"+now.strftime("%Y_%m_%d_%H_%M_%S")

try:
    os.mkdir(path)
except OSError:
    print ("\n Creation of the directory %s failed" % path)
else:
    print ("\n Successfully created the directory %s " % path)

for asset in Assets:
    symbol_name = asset
    
    import yfinance as yf
    
    data = yf.download(symbol_name, period="5y", interval="1wk")
    
    
    for col in data.columns:
        data[col] = pd.to_numeric(data[col],errors='coerce')
    
    data = data.dropna()
    data=data.head(-1)
    
    
    #plt.figure(figsize=(10,4))
    #plt.title(symbol)
    #plt.plot(data['Close'], label='real price')
    #plt.legend()
    #plt.show()
    
    
    df = data
    
    
    # Transform data from non-stationary to stationary
    X = df.Close
    # Method 1: difference data
    stationary = X.diff(1)
    # # Method 2: take the log
    # stationary = np.log(X)
    # # Method 3: take the square root 
    # stationary = np.sqrt(X)
    # # Method 4: take the proprtional change
    # stationary = X.pct_change(1)
    stationary.dropna(axis=0, inplace=True)
    
    # The augmented Dicky-Fuller test - check if stationary
    result = adfuller(stationary)
    # test statistic - more negative means more likely to be stationary
    print('ADF Statistic: %f' % result[0])
    # p-value - reject null hypothesis: non-stationary
    print('p-value: %f' % result[1])
    # critical test statistics - p-values: test statistic for null hypothesis
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # plot stationary dataset
    #stationary.plot(figsize=(10,4))
    #plt.title(symbol_name+' Stationary')
    #plt.show()
    
    
    
    
    
    # Searching over SARIMA model orders
    class Auto_Arima():
        def __init__(self, df, start_p=1, start_q=1, max_p=10, max_q=10,\
            seasonal=False, information_criterion='aic'):
            self.df = df
            self.start_p = start_p
            self.start_q = start_q
            self.max_p = max_p
            self.max_q = max_q
            self.seasonal = seasonal
            self.information_criterion = information_criterion
    
        def arima_results(self):
            results = pm.auto_arima(
                self.df,
                start_p = self.start_p,
                start_q = self.start_q,
                max_p = self.max_p,
                max_q = self.max_q,
                seasonal = self.seasonal,
                # m = 14,
                # D = 1,
                # start_P = 1,
                # start_Q = 1,
                # max_P = 10,
                # max_Q = 10,
                information_criterion = self.information_criterion,
                trace = False,
                error_action = 'ignore',
                suppress_warnings=True,
                stepwise = True,
                scoring = 'mse'
            )
            return results
        
    
    train =df["Close"]
    
    
    arima_model = Auto_Arima(train)
    results = arima_model.arima_results()
    
    
    def one_step_forecast():
        predicted, conf_int = results.predict(n_periods=1, return_conf_int=True)
        return (
            predicted.tolist()[0],
            np.asarray(conf_int).tolist()[0])
    
    predictions = []
    confidence_intervals = []
    
    def plot_train_test(train, df,path, name):
        # Plot the predictions for validation set
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        plt.plot(train, label='train')
        plt.plot(df, label='predicted')
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
    
    for x in range(20):
        predicted, conf = one_step_forecast()
        predictions.append(predicted)
        confidence_intervals.append(conf)
    
        # Updates the existing model
        results.update(predicted)
    
    i = pd.date_range(start = '2020-08-03',freq='W-MON',periods=20)
    # Out-of-sample one-step-forecast based on auto_arima results
    predicted = pd.DataFrame(predictions, index= i, columns=['predicted'])
    
    # Plot real price vs one-step-forecast
    plot_train_test(train, predicted, path, symbol_name)


# print(f'RMSE: {rmse:.2f}')