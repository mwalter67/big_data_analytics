#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

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


# In[9]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso  = Lasso()
parameters = {'alpha':   [0.01,0.02,0.05,0.1,0.2,0.4,0.5,0.6,1,5,8,10,100,199,200,500, 750, 1000],
              'selection': ['random'],
              'max_iter': [10000],
              'normalize': [True],
              'random_state':[1337]}
lasso_regressor = GridSearchCV(lasso, parameters, cv=3)
lasso_regressor.fit(train_features,train_labels)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[10]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()
parameters = {'alpha':   [0.01,0.1,0.5,1,2,3,4,5,6,7,8,10,12,15,20,30],
              'max_iter': [10000],
              'normalize': [True],
              'random_state':[1337]}
ridge_regressor = GridSearchCV(ridge,parameters,cv=3)
ridge_regressor.fit(train_features,train_labels)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

EN  = ElasticNet()
parameters = {'alpha':   [0.01,0.015,0.02,0.05,0.1,0.2,0.4,0.5,0.6,1,5,8,10],
              'l1_ratio': [0.1,0.2,0.4,0.7,0.8],
              'max_iter': [10000],
              'normalize': [True],
              'random_state':[1337],
              'selection': ['random']}
lasso_regressor = GridSearchCV(EN, parameters, cv=3)
lasso_regressor.fit(train_features,train_labels)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[ ]:




