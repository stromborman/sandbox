#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# with open(miles_per_state) as file:
    
# data from https://github.com/fivethirtyeight/data/tree/master/bad-drivers
bad_driv = pd.read_csv('bad-drivers.csv')

# seperating dataframe in accident and insurance information
# all together dropped col 4: perc_not_dist
# since data seems suspect and accurate collection may be difficult

accid = bad_driv.iloc[:,[0,1,2,3,5]].copy()

insur = bad_driv.iloc[:,[0,-2,-1]].copy()


sum_accid = accid.describe()

sns.pairplot(accid)
plt.show()
plt.clf()

accid_coor = accid.corr()
# we only see some weak correlations that are not too suprising
# perc_speed and perc_alch at 0.29 (perhaps alch leads to reckless driving)
# perc_alch and fatal_bmiles at 0.20 (perhaps fatal/mile higher in places where getting cab home harder)
# perc_alch and perc_first at -0.25 (perhaps careful drivers and young drivers)


# compute multilinear regression on fatal_per_bmiles
reg = LinearRegression()
X = accid.iloc[:,2:]
y = accid.iloc[:,1]
reg.fit(X, y)
coef = reg.coef_
# the weak positive correlation between fatal_bmiles and perc_alch is preserved
# with perc_alch coef being 0.19 (the other three < .04 in abs)

# perform a PCA to see if we pick out anything
scaler = StandardScaler()
pca_full = PCA()
X_scaled = scaler.fit_transform(X)
pca_full.fit(X_scaled, y)
pca_trans = pca_full.fit_transform(X_scaled, y)

plt.bar(range(1, pca_full.n_components_ + 1),  pca_full.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])
plt.show()
plt.clf()

# first two components of pca account for 0.79 of variance
two_first_comp_var_exp = pca_full.explained_variance_ratio_[:2].sum()
plt.scatter(pca_trans[:,0], pca_trans[:,1])
plt.show()
plt.clf()

# trying KMeans to cluster states
inertias = []
for k in range(1,10):
    km = KMeans(n_clusters=k, random_state=8)
    km.fit(X_scaled)
    inertias.append(km.inertia_)    
plt.plot(range(1,10), inertias, marker='o')
plt.show()
plt.clf()

# Choosing to use 3 clusters
km = KMeans(n_clusters=3, random_state = 8)
km.fit(X_scaled)

plt.scatter(pca_trans[:,0], pca_trans[:,1], c=km.labels_)
plt.show()
plt.clf()

# violinplot of features grouped by clusters
accid['cluster'] = km.labels_
melt_car = pd.melt(accid, id_vars = ['cluster'], value_vars=list(accid.columns)[2:-1], var_name = 'measurement', value_name = 'percent')
sns.violinplot(x = 'percent', y = 'measurement', hue = 'cluster', data = melt_car)
plt.show()
plt.clf()


# bringing in num of miles driven in each state, to recover total fatalities
miles_driven = pd.read_table('miles_per_state',converters={'state':lambda x:x.strip()})
car_acc_miles = accid.merge(miles_driven, on='state')
car_acc_miles['num_fatal'] = car_acc_miles['fatal_bmiles']*(1/1000)*car_acc_miles['million_miles_annually']


sns.barplot(x='cluster', y='num_fatal', data=car_acc_miles, estimator=sum, ci=None)
plt.show()
plt.clf()

# Calculate the number of states in each cluster and their 'num_fatal' mean and sum.
count_mean_sum = car_acc_miles.groupby(['cluster'])['num_fatal'].agg(['count','mean','sum'])
