#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning the data
"""

import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from fancyimpute import IterativeImputer
from sklearn.preprocessing import StandardScaler


rawtrain = pd.read_csv('exercise_40_train.csv')
rawtest = pd.read_csv('exercise_40_test.csv')
raw = pd.concat({'train':rawtrain, 'test':rawtest})

"""
Basic cleaning/encoding of categorical data
"""

# See what columns have nonnumeric data
raw.columns[raw.dtypes == object]


# These should be numeric data, but had an extra character making them strings
raw['x7'] = raw.x7.str.replace('%','',regex=False).astype('float64')
raw['x19'] = raw.x19.str.replace('$','',regex=False).astype('float64')


# For the remaining, we will count the values and nan in each column.
# We collapse the values into a few bins (to reduce the number of dummy columns we make).
# We bin together values x that have a similar Prob(y|x) (nan is binned this was as well)
# After binning, we then turn the categorical data into numeric data with dummy columns 


# Day of week
raw.x3.value_counts(dropna=False)
# Each day shows up as either its full name or three letter version, so we standardize 
raw.x3 = raw.x3.str[:3]
raw.groupby('x3').mean().y
# Since Fri,Sat,Sun make y more likely to be True, we group them together
def dow_bin(dow):
    if dow in ['Fri','Sat', 'Sun']:
        return 'weekend'
    elif dow in ['Mon', 'Tue', 'Wed', 'Thu']:
        return 'weekday'
raw.x3 = raw.x3.apply(dow_bin)
raw = pd.get_dummies(raw,columns=['x3'], drop_first=True)

X, X_f, y, y_f = train_test_split(df.iloc[:,2:], df.iloc[:,0], test_size=.2) 
# Gender
raw.x24.value_counts(dropna=False)
raw.groupby('x24',dropna=False).mean().y
raw['x24'].fillna('unknown',inplace=True)
raw = pd.get_dummies(raw,columns=['x24'])


raw.x31.value_counts(dropna=False)
raw = pd.get_dummies(raw,columns=['x31'], drop_first=True)


# State
raw.x33.value_counts(dropna=False)
stateres = raw.groupby('x33', dropna=False).mean().y.sort_values()
states = list(stateres.index)
# Bin the high prob together and the low prob together.
def state_bin(state):
    if state in states[:22]:
        return 'L'
    elif state in states[40:]:
        return 'H'
    else:
        return 'M'
raw['x33'] = raw['x33'].apply(state_bin)
raw = pd.get_dummies(raw,columns=['x33'])


# drop this column, constant value
raw.x39.value_counts()
raw.drop(columns=['x39'], inplace = True)


# Month
raw.x60.value_counts(dropna = False)
monres = raw.groupby('x60', dropna=False).mean().y.sort_values()
# Bin the high prob together and the low prob together.
def month_bin(month):
    if month in list(monres.index)[:3]:
        return 'L'
    elif month in list(monres.index)[3:7]:
        return 'M'
    elif month in list(monres.index)[7:]:
        return 'H'    
raw.x60 = raw.x60.apply(month_bin)
raw = pd.get_dummies(raw,columns=['x60'])X, X_f, y, y_f = train_test_split(df.iloc[:,2:], df.iloc[:,0], test_size=.2) 


# Insurance company
raw.x65.value_counts(dropna=False)
raw.groupby('x65', dropna=False).mean().y.sort_values()
raw = pd.get_dummies(raw,columns=['x65'])


# Make of car
raw.x77.value_counts(dropna=False)
raw.groupby('x77', dropna=False).mean().y.sort_values()
# Bin the high prob together and the low prob together.
def make_bin(make):
    if make in ['toyota', 'nissan', 'buick']:
        return 'H'
    elif make in ['ford', 'subaru']:
        return 'M'
    else:
        return 'L'
raw.x77 = raw.x77.apply(make_bin)
raw = pd.get_dummies(raw,columns=['x77'])


raw.x93.value_counts(dropna=False)
raw = pd.get_dummies(raw,columns=['x93'], drop_first=True)


# Only yes or nan, so replace nan with no
raw.x99.value_counts(dropna=False)
raw['x99'].fillna('no', inplace=True)
raw = pd.get_dummies(raw,columns=['x99'], drop_first=True)


"""
Rough feature selection pt.1
"""
# Since there are so many columns, to make doing EDA easier we throw out the
# columns that have a correlation with y that is approximately 0.
# We cut down from 112 to 56 features.

rawtrain = raw.loc['train']
corrdf1 = rawtrain.corrwith(rawtrain['y']).apply(abs).sort_values(ascending=False)
features1 = corrdf1[corrdf1 > .01].index
raw = raw[features1]


"""
Rounding off outliers
"""
# Looking at the histograms of the remaining columns, we see that a handful of
# the numeric columns have outliers.
raw.hist()
plt.show()

# To correct for this, in each column we will round up anything below the 2% and
# similarly will round down anything above the 98%.  clipper is a transformer that will do this.
clipper = FunctionTransformer((lambda x: x.clip(lower=x.quantile(0.02), upper=x.quantile(0.98), axis=1)))

# We calculate the cutoffs only using the training data then adjust all the data 
rawtrain = raw.loc['train']
clipper.fit(rawtrain)
raw = clipper.transform(raw)

# Looking at the histograms for the adjusted data, they look much more reasonable
raw.hist()
plt.show()


"""
Considerations for missing data
"""
# Here we look at the missing data and its relationship to y.  To avoid leakage,
# we just look at the training data to make the decision.
rawtrain = raw.loc['train']

misscol = list(rawtrain.isna().sum()[rawtrain.isna().sum()>0].index)
ycorrmiss = rawtrain.isna()[misscol].corrwith(rawtrain['y'])
ycorrcol = rawtrain[misscol].corrwith(rawtrain['y'])
nummiss = rawtrain.isna().sum()[misscol]
yabscorrcol = ycorrcol.apply(abs)
missdf = pd.concat([nummiss,ycorrmiss, ycorrcol,yabscorrcol],axis=1)
# The dataframe missdf collects together various important stats about the
# numeric features with missing data.

# Since missing data in 'x16','x89','x96' have a non-trivial correlation with y,
# we make new features that mark when this occurs.
importmiss = raw.isna()[['x16','x89','x96']].rename(columns = (lambda x : x+'_nan'))
raw = raw.join(importmiss.astype(int), how='left')

# We note that some features have a large percentage of missing data, but have
# a non-trivial correlation with y (when they do have data).  For this reason
# we will keep the columns.  The hope is that imputation and model specific
# feature selection takes care of them.


"""
Rough feature selction pt.2
"""
# We raise the bar (from .01 to .025) and throw out remaining features with
# a small correlation with y.
# We cut down from 59 to 34 features.

rawtrain = raw.loc['train']
corrdf = rawtrain.corrwith(rawtrain['y']).apply(abs).sort_values(ascending=False)
features = corrdf[corrdf > .025].index


"""
Extra comments
"""
# As a class y is imbalanced in the training dataset so this will need to be accounted
# for in the classification models.  This can be done either by upsampling 
# the cases where y=1 or by weighting them more heavily.  I decided to go
# with the weighting option, mainly because it was simplier to impliment. 







"""
Imputing and scaling the data
"""
# Imputing the missing data.  We then scaling the data, so classification
# models start by viewing/weighting each feature the same.  Fit on the training
# data and then performed on the testing data.
 
steps = [('imputer', IterativeImputer()), ('scaler', StandardScaler())]
datap = Pipeline(steps)

df = raw[features].loc['train']
X = df.iloc[:,1:]
y = df['y']


X_train = pd.DataFrame(datap.fit_transform(X,y),columns=df.iloc[:,1:].columns)    
X_test = pd.DataFrame(datap.transform(raw[features[1:]].loc['test']),columns=df.iloc[:,1:].columns)


"""
Write the adjusted datasets to .csv files
"""
X_train.to_csv('train', index_label=False)
y.to_csv('y', index_label=False)
X_test.to_csv('test', index_label=False)