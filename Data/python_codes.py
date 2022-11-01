## Introduction ##
"""
This code should serve as a good foundation for students, like myself, who are learning to code with python. 
Many of these mechanisms are simple but helpful when dealing with problems of greater complexity and, as such, serve as a helpful baseline. 
"""

## Importing Libraries used in various examples ##

"""
When running into difficulties with specific undertakings, it is often necessary to visit Github or other websites for examples.
Many of the examples cited the below python packages and as such I have included them all. 
"""

import pandas as pd
import multiprocessing
import numpy as np
import re 
import hashlib
import os 
import subprocess
import time
import xarray as xr
import sys
import random
import shap
import warnings
import pydot
import math
from tabulate import tabulate
from IPython.display import display
import numpy

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter

from scipy.stats import mstats
import scipy.stats as st

from operator import itemgetter

from sklearn.metrics import  r2_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn import svm as sk_svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression as lm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
  
    
import matplotlib 
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

## Setting a directory ##

main_directory = "C:/Users/dylan/..../...../...."


###########################################################################

## Viewing Options ## 

"""
These configurations allow for easier oversight of the massive dataset with which Bluwstein, Buckmann, Joseph, Kapadia, and Şimşek were working. 
"""

pd.set_option("max_rows", None)
pd.set_option('display.max_columns', None)

# To read the excel file from the Jordà-Schularick-Taylor Macrohistory Database. 

df = pd.read_csv(r'C:\Users\dylan\...\...\JSTdatasetR6.csv')

###########################################################################

## Renaming Variables ## 

df.rename(columns={
        "crisisJST": "crisis",
        'stir': 'short_rate',
        'ltrate': 'long_rate',
        'iy': 'invgdp_ratio',
        'debtgdp': 'pdebtgdp_ratio',
        'money': 'broad_money',
        'narrowm': 'narrow_money',
        'tloans': 'total_loans',
        'tbus': 'business_loans',
        'thh': 'household_loans',
        'tmort': 'mortgage_loans',
        #this one is not in the dataset 'stocks': 'stock',
        'hpnom': 'hp',
        'rconsbarro': 'real_consumption'
    }, inplace=True)  

###########################################################################

## Removing Crisis Years Manually ##

"""
The Authors deliberately remove years of extreme circumstance as these would obstruct the predictive model in effectively recognizing conventional signals of a crisis.
The world wars and the great depression are outside the scope of normality. 
"""

years_ex = list(range(1933,1946)) + list(range(1914, 1919))
df = df[~df.year.isin(years_ex)]

###########################################################################

## Generating Variables ##

# Computing certain variables as described by authors. 

# Rate differential or yield curve
df.loc[:,'rate_differential'] = df['long_rate'] - df['short_rate']

# Public debt from public debt / gdp ratio
df.loc[:,'public_debt'] = df['pdebtgdp_ratio'] * df['gdp']

# Computing Investment from investment / gdp ratio
df.loc[:,'inv'] = df['invgdp_ratio'] * df['gdp']

# Calculate debt to service ratios
df.loc[:,'debt_service_ratio'] = df['total_loans'] * df['long_rate']  / 100

## Setting Variables in terms of GDP ##

"""
In order to have a common denominator, the authors have set various variables in terms of GDP.
"""

pre_gdp_ratios = ['broad_money', 'narrow_money', 'total_loans', 'business_loans', 'household_loans', 'mortgage_loans', 'ca', 'cpi', 'debt_service_ratio', 'inv', 'public_debt']

def make_ratio(data_input, variables, denominator="gdp"):
    #computes the ratio of two variables. By default the denominator is gdp.
    
    names_out = []
    if isinstance(variables,str):
        variables = [variables]
    data = data_input.copy()
    for var in variables:
        varname = var + '_' + denominator
        data[varname] = data[var] / data[denominator]
        names_out.append(varname)
    return data, names_out 

df, gdp_ratios = make_ratio(df, pre_gdp_ratios, denominator='gdp')

# Global slope of the yield curve

for year in df["year"].unique():
    ix = df["year"] == year
    for country in df["iso"].unique():
        #computing the average across all countries but the selected one
        perc_pos = df.loc[ix.values & (df.iso != country).values, "rate_differential"].mean()
        
        if not np.isnan(perc_pos):
            df.loc[ix.values & (df.iso == country).values, "global_rate_differential"] = perc_pos 
                   
# Generate Log of GDP

df['ln_gdp'] = np.log(df['gdp']) 

###########################################################################

# Dropping certain variables for easier data manipulation. 

df.drop(columns=['rent_ipolated','bond_tr',
                 'housing_tr','housing_rent_yd','eq_capgain_interp','eq_tr_interp',
                 'eq_dp_interp','housing_capgain_ipolated', 'peg','JSTtrilemmaIV',
                 'peg_strict', 'peg_type','peg_base',
                 'capital_tr','risky_tr', 'safe_tr', 'crisisJST_old'], inplace=True)

###########################################################################

"""
In addition to dropping years of extreme circumstance, the authors remove observations with any missings in order to have a strongly balanced dataset. 

"""

# Dropping observations with missings from choice variables manually 

df = df[df['short_rate'].notna()]
df = df[df['long_rate'].notna()]
df = df[df['invgdp_ratio'].notna()]
df = df[df['pdebtgdp_ratio'].notna()]
df = df[df['broad_money'].notna()]
df = df[df['total_loans'].notna()]
df = df[df['business_loans'].notna()]
df = df[df['household_loans'].notna()]
df = df[df['mortgage_loans'].notna()]
df = df[df['hp'].notna()]
df = df[df['real_consumption'].notna()]
df = df[df['ca'].notna()]
df = df[df['cpi'].notna()]
df = df[df['unemp'].notna()]

###########################################################################

# Crisis Variable 

"""
The authors have deliberately marked two years before actual crises within the dataset as crises and removing four years suceeding any crisis event. 
The first omission serves to remove years of the economy that are affected by crisis dynamics and, as such, could dilute
the predictive capability of the forecasting models. 

The second change is to ensure that macroprudential policy makers have enough time to implement potential changes. The authors have also
carefully ensured that a minimum number of crises would still exist within the training set and nested cross validation folds. 

I had many issues implementing the above codes. To proceed with my replication,
my analysis became substantially simpler than the work of Bluwstein, Buckmann, Joseph, Kapadia, and Şimşek (2021). 

Rather than training the machine learning model on predicting true financial crises, I sought to experiment with the ability of the machine learning mechanism to recognize
contractionary movements in the economy. 

In this replication, rather than employ the binary crisis variable, I will follow the work of Céline Carrère and Maria Masood (2018). 

For a copy of their work, please see: https://onlinelibrary.wiley.com/doi/abs/10.1111/twec.12646

a binary contraction variable to account for states of weaker economic performance.
Instead of requiring at least one negative standard deviation away from the trend level of the natural log of GDP, as Carrère and Masood have done, for simplicity,
any negative value returned by a Hodrick-Prescott filter is marked positive by the binary contraction variable.

Thus, I could examine which variables have the most predictive strength in outlining economic contractions.


"""
# Setting the Hodrick Prescott Filter

gdp_cycle,gdp_trend = hpfilter(df['ln_gdp'], lamb=1600)
# Lamb = 1600 for quarterly data
# Lamb = 6.25 for yearly

df["ln_gdp_trend"] = gdp_trend

df["ln_gdp_cycle"] = gdp_cycle

df["contraction"] = df['ln_gdp_cycle'] < 0

df["contraction_dummy"] = df["contraction"].astype(int)

###########################################################################

"""
Python has difficulty reading string elements so I converted the countries in the dataset to a numeric list to proceed, based on their identity codes. 

"""
status = [193,124,156,128,172,132,134,178,136,158,138,142,182,184,144,146,112,111]
identity_code = [1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,17,18]

df['country'] = df['ifs'].replace(status, identity_code)

"""
Country     ifs  Identity_code
Australia   193      1
Belgium     124      2
Canada      156      3
Denmark     128      4
Finland     172      5
France      132      6
Germany     134      7
Ireland     178      8
Italy       136      9
Japan       158      10
Netherlands 138      11
Norway      142      12
Portugal    182      13
Spain       184      14
Sweden      144      15
Switzerland 146      16
UK          112      17
US          111      18

"""
#dropping more variables because of an Nan error message to have a perfectly balanced dataset
df.drop(columns=['iso', 'ifs', 'contraction', 'eq_tr', 'bill_rate', 'housing_capgain', 'housing_rent_rtn', 'eq_capgain','eq_dp','bond_rate','eq_div_rtn', 'lev', 'noncore'], inplace=True)

###########################################################################

# Checks to see if the data is strongly balanced. 

# To check if they are nans
pd.isna(train_features).any()
# If true, then there are nans in the dataset

# Checking for nan
np.isnan(df)

# Coordinates of nan
np.where(np.isnan(df))

# Replace nan with zero and infinite numbers with finite numbers
np.nan_to_num(df)

# Checking for nans in training _ features #
pd.isna(train_features).any()

# Dropping the nas
df = df.dropna()

###########################################################################

# Setting up a training and test set. 

"""
The below code is to create a training and test set. I had difficulties with this process but the below seems to work. 

"""
# One hot encoding
features = pd.get_dummies(df)

# Extract features and labels
labels = features['contraction_dummy']
features = features.drop('contraction_dummy', axis=1)

# List of features for later use 
feature_list = list(features.columns)


# Converting to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

###########################################################################

# Check the training and test set. 

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

###########################################################################

# Baseline predictions are historial averages 
baseline_preds = test_features[:, feature_list.index('rate_differential')]

# Baseline Errors 
baseline_errors = abs(baseline_preds - test_labels)
print('Average Baseline Error: ', round(np.mean(baseline_errors), 2), 'contractions')


###########################################################################

# Correlation Matrix

# Setting the size of the figure
f, ax = plt.subplots(figsize=(10, 8))

# Finding the correlation
corr = df.corr()

# Plotting the correlation
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

###########################################################################

# Another attempt at training and test set

"""
As stated, I ran into issues working with a training and test dataset. The following code consists of further examples found online, afterwhich I was able
to somewhat generate some results.
"""
# Splitting the dataset
Inputs = df.drop('contraction_dummy', axis=1)
output = df['contraction_dummy']

# Splitting the dataset 
X_train, X_test, y_train, y_test=train_test_split(Inputs, output, test_size=0.3)

# From sklearn ExtraTreesRegressor
ET_regressor = ExtraTreesRegressor()

print(ET_regressor.get_params())

# Train the model
ET_regressor.fit(X_train, y_train)

# Making predictions
Regressor_pred = ET_regressor.predict(X_test)

# Fitting the size of the plot
plt.figure(figsize=(15, 8))

# Plotting the graphs
plt.plot([i for i in range(len(y_test))],y_test, color = 'red',label="actual values")
plt.plot([i for i in range(len(y_test))],Regressor_pred, color='blue', label="Predicted values")

# Display
plt.legend()
plt.show()

# Evaluating model performance
print('R-square score is :', r2_score(y_test, Regressor_pred))

###########################################################################

# Extremely Randomized Trees Confusion Matrix

# initializing the model
ET_classifier = ExtraTreesClassifier()

# Training the model
ET_classifier.fit(X_train, y_train)

# making predictions
classifier_pred = ET_classifier.predict(X_test)

# Creating a Confusion Matrix

# Providing actual and predicted values
cm = confusion_matrix(y_test, classifier_pred)

# If True, write the data value in each cell
sns.heatmap(cm,annot=True)

#accuracy score
accuracy_score(y_test,y_pred)

###########################################################################

# Random Forest Classifier Confusion Matrix

# Random forest vs random tree classifier
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Testing the model
y_pred = classifier.predict(X_test)

# Providing actual and predicted values
cm = confusion_matrix(y_test, y_pred)

# If True, write the data value in each cell
sns.heatmap(cm,annot=True)

# Printing the accuracy of the model
print(accuracy_score(y_test, y_pred))

###########################################################################
