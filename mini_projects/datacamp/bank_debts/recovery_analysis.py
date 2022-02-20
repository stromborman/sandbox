#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import statsmodels.api as sm
from scipy import stats
# import numpy as np
import pandas as pd

df = pd.read_csv('bank_data.csv', index_col='id', usecols=[1,2,3,4,5,6])
df['recovery_strategy'] = df['recovery_strategy'].str.extract(r'(\d)').astype(int)

"""
The bank has implemented different recovery strategies at different thresholds
[1000, 2000, 3000, 5000] where the greater the Expected Recovery Amount, 
the more effort the bank puts into contacting the customer.

What follows is an investigation of how effective each higher level strategy
is compared to the previous one by investigating jumps in the actual
recovery amounts when the expected recovery amounts are near the thresholds.
"""


thresholds = [1000,2000,3000,5000]

def analysis(level, window=50):
    threshold = thresholds[level]
    df['constant'] = 1
    df1 = df.loc[(df['expected_recovery_amount'] < threshold + window*(level+1)) &\
                 (df['expected_recovery_amount'] >= threshold - window*(level+1))]
            
    low = df1.loc[df['recovery_strategy']==level]
    high = df1.loc[df['recovery_strategy']==level+1]
        
    _, p_val_age = stats.kruskal(low['age'],high['age'])
    
    cross = pd.crosstab(df1.recovery_strategy, df1.sex)
    
    chi2_stat, p_val_sex, dof, ex = stats.chi2_contingency(cross)
        
    _, p_val_act = stats.kruskal(low['actual_recovery_amount'],high['actual_recovery_amount'])
    
    pval_dist = min([p_val_age, p_val_sex, p_val_act])
    
    X = df1[['constant','expected_recovery_amount', 'recovery_strategy']]
    # X = sm.add_constant(X)
    y = df1.actual_recovery_amount
    results = sm.OLS(y,X).fit()
    
    return int(results.params[2]), round(results.pvalues[2],3), round(pval_dist, 3), results

final = pd.DataFrame()
final['recovery_delta'] = [analysis(i)[0] for i in range(4)]
final['delta_p'] = [analysis(i)[1] for i in range(4)]
final['delta_dist'] = [analysis(i)[2] for i in range(4)]

# The bump in thresholds is certainly effective at the 0, 1, and 2 levels,
# with returns experiencing statistical significant deltas of 230, 568, 1686.
# The negative(!) effect size at level 3 can be chalked up to the OLS model no longer
# being suitable in this range.
    