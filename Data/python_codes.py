## Introduction ##
"""
This code should serve as a good foundation for students, like myself, who are learning to code with python. 
Many of these mechanisms are simple but helpful when dealing with problems of greater complexity and, as such, can serve as a helpful baseline. 
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
The Authors deliberately have remove extreme case years as these would obfuscate the objective of the machine learning mechanism, namely accurately predicting
when financial crises occur. The world wars and the great depression are outside the scope of normality. 
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









