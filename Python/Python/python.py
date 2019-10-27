#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import random 
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from pandas.plotting import table

import warnings
warnings.filterwarnings("ignore")


# In[7]:


df = pd.read_csv("nyc-rolling-sales.csv")

df.columns = ['Unnamed: 0', 'borough', 'neighborhood','building_class category','tax_class_at_present', 'block', 'lot', 'ease_ment','building_class_at_present', 'address', 'apartment_number', 'zip_code',
       'residential_units', 'commercial_units', 'total_units','land_square_feet', 'gross_square_feet', 'year_built','tax_class_at_time_of_sale', 'building_class_at_time_of_sale',
       'sale_price', 'sale_date']


# deleting the Unnamed column
del df['Unnamed: 0']
del df['ease_ment']
del df['address']
del df['zip_code']

df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['land_square_feet'] = pd.to_numeric(df['land_square_feet'], errors='coerce')
df['gross_square_feet']= pd.to_numeric(df['gross_square_feet'], errors='coerce')

# Both TAX CLASS attributes should be categorical
df['tax_class_at_time_of_sale'] = df['tax_class_at_time_of_sale'].astype('category')
df['tax_class_at_present'] = df['tax_class_at_present'].astype('category')

#SALE DATE is object but should be datetime
df['sale_date']    = pd.to_datetime(df['sale_date'], errors='coerce')
df['sale_year']    = df['sale_date'].dt.year


#Replacing borough by their name
# Renaming BOROUGHS
df['borough'][df['borough'] == 1] = 'Manhattan'
df['borough'][df['borough'] == 2] = 'Bronx'
df['borough'][df['borough'] == 3] = 'Brooklyn'
df['borough'][df['borough'] == 4] = 'Queens'
df['borough'][df['borough'] == 5] = 'Staten_Island'


# In[8]:


print(df['land_square_feet'].corr(df['gross_square_feet']))


# In[9]:


duplicateRowsDF = df[df.duplicated()]
df = df.drop_duplicates(df.columns, keep='last')
print("Number of duplicates in the given dataset after cleanup = {0}".format(sum(df.duplicated(df.columns))))


# In[10]:


# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct_missing = 100 - np.round(100 * pd.Series(count) / len(df), 2)

schön = [go.Bar(
            y= df.columns,
            x = count_pct_missing,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]

layout = go.Layout(
    title='Percentage of missing values values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)
fig = go.Figure(data=schön, layout = layout)
py.iplot(fig, filename='barplottype')


# In[11]:


# as there are rows where sale price is null, we should remove them from our dataset
df = df[df['sale_price'] > 0]
df = df[df['land_square_feet'] > 0]
df = df[df['gross_square_feet'] > 0]
df = df[df['year_built'] > 0]

df['age_of_building'] = df['sale_year'] - df['year_built']



del df['sale_year']
del df['year_built']
del df['sale_date']
del df['apartment_number']



print("Length of dataset after cleanup = {0}".format(len(df)))

print(df.describe())


# In[12]:


print("Length of dataset after cleanup = {0}".format(len(df)))
print(df.describe())


# In[13]:


# lets find out the percentage of non null values in each column and plot them to get a better view
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct_missing = 100 - np.round(100 * pd.Series(count) / len(df), 2)

schön = [go.Bar(
            y= df.columns,
            x = count_pct_missing,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]

layout = go.Layout(
    title='Percentage of missing values values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)
fig = go.Figure(data=schön, layout = layout)
py.iplot(fig, filename='barplottype')


# In[14]:


plt.figure(figsize=(10,6))
sns.distplot(df['sale_price'])
plt.title('Histogram of Sale Price in USD')
plt.ylabel('Normed Frequency')
plt.show()
len(df)

plt.figure(figsize=(10,6))
sns.distplot(df['sale_price'].apply(np.log))
plt.title('Histogram of Sale Price in USD')
plt.ylabel('Normed Frequency')
plt.show()

#Deeds, transfer from parents to children that is why we have 0 in sale price.


# In[15]:


print(df.quantile(0.95))
print(df.quantile(0.05))
print(df.describe())


# In[16]:


print(len(df))
print(df.describe())
print(df.quantile(0.95))
print(df.quantile(0.05))

plt.figure(figsize=(10,6))
sns.distplot(df['sale_price'])
plt.title('Histogram of Sale Price in USD')
plt.ylabel('Normed Frequency')
plt.show()
len(df)

#Deeds, transfer from parents to children that is why we have 0 in sale price.


# In[18]:


df_cheap = df[df["sale_price"] < 125000]
print(len(df_cheap))

df_expensive = df[df["sale_price"] > 3500000]
print(len(df_expensive))

df = df[(df["sale_price"] > 125000) & (df["sale_price"] < 3500000)]
print(len(df))

print(df.describe())
print(df.quantile(0.95))
print(df.quantile(0.05))

print(df.info())


df.to_csv(r'data_clean.csv')


# In[19]:


plt.figure(figsize=(10,6))

sns.boxplot(x='sale_price', data=df)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of Sale Price in USD')
plt.show()


# In[20]:


plt.figure(figsize=(10,6))

sns.distplot(df['sale_price'])
plt.title('Histogram of Sale Price in USD')
plt.ylabel('Normed Frequency')
plt.show()


# In[21]:


# We got rid of Null values for the Sale Price variable
variables = df.columns
count = []

for variable in variables:
    length = df[variable].count()
    count.append(length)
    
count_pct_missing = 100 - np.round(100 * pd.Series(count) / len(df), 2)

schön = [go.Bar(
            y= df.columns,
            x = count_pct_missing,
            width = 0.7,
            opacity=0.6, 
            orientation = 'h',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5,
                )
            ),
        )]

layout = go.Layout(
    title='Percentage of missing values values in each column',
    autosize = False,
    width=800,
    height=800,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
)
fig = go.Figure(data=schön, layout = layout)
py.iplot(fig, filename='barplottype')


# In[22]:


plt.figure(figsize=(10,6))

corr_df = df[['sale_price', 'total_units','gross_square_feet',  'land_square_feet', 'residential_units', 
         'commercial_units', 'age_of_building']]

corr = corr_df.corr()
corr.style.background_gradient(cmap='RdBu_r').set_precision(2)

# 'RdBu_r' & 'BrBG' 'coolwarm' are other good diverging colormaps


# In[23]:


df.to_csv(r'data_clean.csv')


# In[24]:


import numpy as np

plt.figure(figsize=(10,6))

sns.distplot(df['sale_price'].apply(np.log))
plt.title('Histogram of Sale Price in USD')
plt.ylabel('Normed Frequency')
plt.show()


# In[28]:


plt.figure(figsize=(10,6))
print(df["borough"].value_counts())
explode = (0.3, 0.1, 0.1, 0.2 , 0.1)  
labels = ["Queens","Brooklyn","Staten Island","Bronx","Manhattan"]


#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05,0.05)
 
plt.pie(df["borough"].value_counts(), colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = (0.05, 0.05, 0.05, 0.05,0.05))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# In[29]:


plt.figure(figsize=(10,6))
classi = df["tax_class_at_time_of_sale"].value_counts()
del classi[3]
print(classi)
labels = ["Class 1","Class 2","Class 3"]
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05,0.05)
 
plt.pie(classi, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = (0.05, 0.05, 0.05))
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


