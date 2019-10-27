#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_clean.csv")
del df["Unnamed: 0"]

df = df[["gross_square_feet","block","land_square_feet","lot","age_of_building","borough","residential_units","commercial_units","total_units","sale_price"]]

df['borough'] = df['borough'].astype('category')


X, y = df.iloc[:,:-1],df.iloc[:,-1]

one_hot_encoded_X = pd.get_dummies(X)
print("# of columns after one-hot encoding: {0}".format(len(one_hot_encoded_X.columns)))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_X, y, test_size=0.25, random_state=1337)


# In[2]:


xgb = XGBRegressor()

parameters = {
 'n_estimators': [500,1000,2000],
 'max_depth': [3,5,7,10],
 'min_child_weight': [1,3,5,10],
 'learning_rate': [0.01, 0.05, 0.075, 0.1, 0.15],
 'eval_metric': ['rmse'],
 'seed': [1337],
 'objective': ['reg:squarederror'],
 'nthread' : [6],
 'njobs': [4],
}


xgb_regressor = GridSearchCV(xgb,parameters,cv=3)
xgb_regressor = xgb_regressor.fit(X_train, y_train,
                                  early_stopping_rounds=10,
                                  eval_set=[(X_test, y_test)],
                                  verbose=False)

# summarize results
print("Best: %f using %s" % (xgb_regressor.best_score_, xgb_regressor.best_params_))

