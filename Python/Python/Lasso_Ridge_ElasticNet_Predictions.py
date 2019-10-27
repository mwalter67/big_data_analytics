#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("df.csv")
del df["Unnamed: 0"]
df = df.rename({'1200': 'Sale Price'}, axis='columns')

features = pd.get_dummies(df)
labels = np.array(features['Sale Price'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Sale Price', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 1337)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[21]:


lasso = Lasso(alpha=199, max_iter=10000, selection = 'random', normalize = True, random_state = 1337)
lasso.fit(train_features,train_labels)
train_score=lasso.score(train_features,train_labels)
test_score=lasso.score(test_features,test_labels)
coeff_used = np.sum(lasso.coef_!=0)

print("number of features used:", coeff_used)
predictions_lasso = lasso.predict(test_features)


rmse_lasso = sqrt(mean_squared_error(predictions_lasso,test_labels))
print("RMSE:", round(rmse_lasso,2))


# In[22]:


ridge = Ridge(alpha=3, max_iter=10000, normalize = True, random_state = 1337)
ridge.fit(train_features,train_labels)
train_score=ridge.score(train_features,train_labels)
test_score=ridge.score(test_features,test_labels)
coeff_used = np.sum(ridge.coef_!=0)

print("number of features used:", coeff_used)
predictions_ridge = ridge.predict(test_features)

rmse_ridge = sqrt(mean_squared_error(predictions_ridge,test_labels))
print("RMSE:", round(rmse_ridge,2))


# In[23]:


EN = ElasticNet(alpha=0.01, max_iter=10000, normalize=True, l1_ratio = 0.8)
EN.fit(train_features,train_labels)
train_EN=EN.score(train_features,train_labels)
test_EN=EN.score(test_features,test_labels)
coeff_used = np.sum(EN.coef_!=0)

print("number of features used:", coeff_used)

predictions_EN = EN.predict(test_features)
rmse_EN = sqrt(mean_squared_error(predictions_EN,test_labels))
print("RMSE:", round(rmse_EN,2))


# In[24]:


df = pd.read_csv("data_clean.csv")
del df["Unnamed: 0"]

df = df[["gross_square_feet","block","land_square_feet","lot","age_of_building","borough","residential_units","commercial_units","total_units","sale_price"]]
df['borough'] = df['borough'].astype('category')

features = pd.get_dummies(df)
labels = np.array(features['sale_price'])

features= features.drop('sale_price', axis = 1)
feature_list = list(features.columns)
features = np.array(features)


train_features_rf, test_features_rf, train_labels_rf, test_labels_rf = train_test_split(features, labels, test_size = 0.25, random_state = 1337)

print('Training Features Shape:', train_features_rf.shape)
print('Training Labels Shape:', train_labels_rf.shape)
print('Testing Features Shape:', test_features_rf.shape)
print('Testing Labels Shape:', test_labels_rf.shape)

rf = RandomForestRegressor(n_estimators = 1200,
                           min_samples_split = 10,
                           min_samples_leaf = 2,
                           max_features = 'auto',
                           max_depth = 40,
                           bootstrap = True,
                           random_state = 1337)
rf.fit(train_features_rf, train_labels_rf)
predictions_rf = rf.predict(test_features_rf)


rmse_rf = sqrt(mean_squared_error(predictions_rf,test_labels_rf))
print("RMSE:", round(rmse_rf,2))


# In[25]:


def perf(rmse,rmse_rf):   
    ecart = ((rmse - rmse_rf)/(rmse_rf))*100
    ecart = 100 - round(ecart,2)
    print("Le Random Forest est plus performant de " + str(ecart) + "%")

def r2_score_(predictions,test_labels):
    r2_score_rf = round(r2_score(predictions,test_labels),3)
    print("Le R^2 est de: " + str(r2_score_rf))

def MAE(predictions,test_labels):
    MAE = round(mean_absolute_error(predictions,test_labels),3)
    print("Le MAE est de : " + str(MAE))


# In[26]:


perf(rmse_lasso,rmse_rf)
perf(rmse_ridge,rmse_rf)
perf(rmse_EN,rmse_rf)


# In[27]:


r2_score_(predictions_rf,test_labels_rf)
r2_score_(predictions_lasso,test_labels)
r2_score_(predictions_ridge,test_labels)
r2_score_(predictions_EN,test_labels)


# In[28]:


MAE(predictions_rf,test_labels_rf)
MAE(predictions_lasso,test_labels)
MAE(predictions_ridge,test_labels)
MAE(predictions_EN,test_labels)

