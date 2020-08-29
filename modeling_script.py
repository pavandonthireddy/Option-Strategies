# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:02:04 2020

@author: Pavan
"""

import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# User-defined model.
# A user-defined wrapper function for model training that takes the following
# arguments: (1) a horizon-specific pandas DataFrame made with create_lagged_df(..., type = "train")
# (e.g., my_lagged_df$horizon_h)
def py_model_function(data):
  
  X = data.iloc[:, 1:]
  y = data.iloc[:, 0]
  
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  model_lasso = linear_model.Lasso(alpha = 0.1)
  
  model_lasso.fit(X = X, y = y)
  
  return({'model': model_lasso, 'scaler': scaler})

# User-defined prediction function.
# The predict() wrapper function takes 2 positional arguments. First,
# the returned model from the user-defined modeling function (py_model_function() above).
# Second, a pandas DataFrame of model features. For numeric outcomes, the function 
# can return a 1- or 3-column pandas DataFrame with either (a) point
# forecasts or (b) point forecasts plus lower and upper forecast bounds (column order and names do not matter).
def py_prediction_function(model_list, data_x):
  
  data_x = model_list['scaler'].transform(data_x)
  
  data_pred = pd.DataFrame({'y_pred': model_list['model'].predict(data_x)})
  
  return(data_pred)