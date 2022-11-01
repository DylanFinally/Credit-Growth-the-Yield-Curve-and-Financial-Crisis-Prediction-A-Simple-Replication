# This code should serve as a good foundation for students, like myself, who are learning to code with python. 
# Many of these mechanisms are simple but helpful when dealing with problems of greater complexity and, as such, can serve as a helpful baseline. 

## Importing Libraries used in various examples ##
# When running into difficulties with specific undertakings, it is often necessary to visit Github or other websites for examples.
# Many of the examples cited the below python packages and as such I have included them all. 

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

## Viewing Options ## 

# These configurations allow for easier oversight of the massive dataset with which Bluwstein, Buckmann, Joseph, Kapadia, and Şimşek were working. 

pd.set_option("max_rows", None)
pd.set_option('display.max_columns', None)

# To read the excel file from the Jordà-Schularick-Taylor Macrohistory Database. 

df = pd.read_csv(r'C:\Users\dylan\...\...\JSTdatasetR6.csv')

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

# Removing Crisis Years Manually 


