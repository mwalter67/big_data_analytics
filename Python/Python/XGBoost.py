#!/usr/bin/env python
# coding: utf-8

# In[18]:


import xgboost as xgb
from sklearn.tree import export_graphviz

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

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


# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_X, y, test_size=0.25, random_state=1337)


# In[20]:


from xgboost import XGBRegressor

print(np.shape(X_train), np.shape(X_test))

xg_model = XGBRegressor(n_estimators=500,
                        learning_rate=0.075,
                        max_depth = 7,
                        min_child_weight = 5,
                        eval_metric = 'rmse',
                        seed = 1337,
                        objective = 'reg:squarederror')
xg_model.fit(X_train, y_train, early_stopping_rounds=10,
             eval_set=[(X_test, y_test)], verbose=False)
predictions = xg_model.predict(X_test)

max_estimators = len(xg_model.evals_result()['validation_0']['rmse'])
print(max_estimators)
max_estim_rmse = pd.DataFrame(xg_model.evals_result()['validation_0']['rmse'], columns=['rmse'])
plt.plot(max_estim_rmse)
plt.ylabel("RMSE")
plt.xlabel("Max Estimators")
xgb.plot_importance(xg_model)
plt.show()


# In[21]:


rmse_rf = sqrt(mean_squared_error(predictions,y_test))
print("RMSE:", round(rmse_rf,2))


# In[22]:



from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(round(mean_absolute_error(predictions, y_test))),2)


# In[23]:


def r2_score_(predictions,test_labels):
    r2_score_rf = round(r2_score(predictions,test_labels),3)
    print("Le R^2 est de: " + str(r2_score_rf))


# In[24]:


r2_score_(predictions,y_test)

