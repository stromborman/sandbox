#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logisitic Regression
"""
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.ensemble import GradientBoostingClassifier

X = pd.read_csv('train')
y = pd.read_csv('y').iloc[:,0]
X_test = pd.read_csv('test')

# The basic LogReg model we will use.  The class_weight option accounts
# for imbalance in the distribution of y in the training data.
logreg = LogisticRegression(class_weight='balanced')

# Feature selection for logistic regression, via recursive feature elimination.
# We want to reduce the number of features we use to prevent overfitting. 
rfe=RFE(logreg)
hyper_params = [{'n_features_to_select': list(range(1, 25))}]

# For setting up cross-validation, we use Stratified to account for the imbalence in the data.
folds = StratifiedKFold(n_splits = 5, shuffle = True)
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'roc_auc', 
                        cv = folds,
                        return_train_score=True)      
model_cv.fit(X, y) 

# cv_results is the dataframe showing the results of using different numbers of features.
cv_results = pd.DataFrame(model_cv.cv_results_)

# plot showing how the mean auc test score changes with the number of features.
plt.figure(figsize=(10,10))
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('auc')
plt.legend(['test score', 'train score'], loc='upper left')


# Rerun the proceedure, to actually get the names of the best features
rfef = RFE(logreg, n_features_to_select=17)
rfef.fit(X, y)

# The features to keep
pd.Series(list(X.columns[rfef.support_])).to_csv('feat', index_label=False)
feat = list(pd.read_csv('feat').iloc[:,0])

# Finally we do a search for the best regularization constant
logregcv = LogisticRegressionCV(class_weight='balanced',scoring='roc_auc')
logregcv.fit(X[feat],y)

# The best regularization constant, which is 4th in the list of C's tried
logregcv.C_

# The mean of the auc for the 5 folds with best C 
logregcv.scores_[1.0][:,3].mean()
# 0.7626295703727779

# writing the predictions
with open('logregprob', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(list(logregcv.predict_proba(X_test[feat])[:,1]))
    

"""
Gradient Boosting
"""
# Run a 5-fold cross validation for Gradient Boosting
gbc_cv = cross_validate(GradientBoostingClassifier(), X[feat], y, scoring='roc_auc')

# The mean auc score for the 5 folds
gbc_cv['test_score'].mean()
# 0.7546036023028705

# Fitting the GradBooster
gbc = GradientBoostingClassifier()
gbc.fit(X[feat], y)

# writing the predicitions
with open('gradboostprob', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(list(logregcv.predict_proba(X_test[feat])[:,1]))