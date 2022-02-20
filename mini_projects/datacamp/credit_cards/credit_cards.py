#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic exercise of cleaning data, normalizing it, and running a logicical regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss


cc_apps = pd.read_csv('cc_approvals.data', header=None)

"""
Preprocessing data
"""
# Columns are:
# Gender 0, Age 1, Debt 2, Married 3, BankCustomer 4, EducationLevel 5, 
# Ethnicity 6, YearsEmployed 7, PriorDefault 8, Employed 9, CreditScore 10, 
# DriversLicense 11, Citizen 12, ZipCode 13, Income 14,
# and finally the ApprovalStatus 15

# missing data is represented with ?
cc_apps = cc_apps.replace('?', np.NaN)

# columns 2, 7, 10, 14 are marked as numeric data
#cc_apps.info()

# however age, column 1 is also numeric
cc_apps = cc_apps.astype({1:float})

# drop zip_code (for now)
cc_apps = cc_apps.drop([13], axis=1)

# rearrange columns
cc = cc_apps[[1, 2, 7, 10, 14, 0, 3, 4, 5, 6, 8, 9, 11, 12, 15]]

numeric = [1,2,7,10,14]
categorical = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]

# only a handful of missing data
cc.isnull().sum()

# for numeric data impute with mean
# for categorical data impute with most frequent
for col in cc.columns:
    if col in numeric:
        cc[col] = cc[col].fillna(cc[col].mean())
    if col in categorical:
        cc[col] = cc[col].fillna(cc[col].value_counts().index[0])
        
# convert categorical data via get_dummies 
# (category with 3 values becomes 3 binary columns)
cc_dum = pd.get_dummies(cc, columns = categorical).drop(columns=['15_-'])


"""
Splitting data and normalizing
"""

cc = cc_dum.to_numpy()

# Segregate features and labels into separate variables
X,y = cc[:,0:-1] , cc[:,-1]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33,random_state=123)

scaler = QuantileTransformer()
scaler.fit(X_train[:,:5])

sX_train = np.concatenate((scaler.transform(X_train[:,:5]),X_train[:,5:]), axis = 1)
sX_test = np.concatenate((scaler.transform(X_test[:,:5]),X_test[:,5:]), axis = 1)

"""
Basic logistic regression
"""

logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(sX_train, y_train)

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(sX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(sX_test, y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred)

# probability predictions
y_prob = logreg.predict_proba(sX_test)

# logloss score
log_loss(y_test, y_prob)