#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:18:08 2022

@author: cadoi
"""

import pandas as pd
import numpy as np

rep = pd.read_csv('~/advent2021/input3', squeeze = True, dtype = str, header=None)

df = np.full((1000,12), 0)
df = pd.DataFrame(df)
    
for j in range(0,1000):
    for i in range(0,12):
        df[i][j] = rep[j][i]
        
        
gam_bin = np.array([df[i].value_counts().index[0] for i in range(0,12)])
eps_bin = np.array([df[i].value_counts().index[1] for i in range(0,12)])

gam = gam_bin @ (2**np.arange(12)[::-1]) 
eps = eps_bin @ (2**np.arange(12)[::-1])
power = gam*eps

print('Answer to day3/part1: power= '+str(power)) 

F = np.empty(13, dtype=pd.DataFrame)
F[0] = df
for i in range(0,12):
    if F[i][i].mean() >= .5:
        n = 1
    else:
        n = 0
    F[i+1] = F[i][F[i][i]==n]
    
G = np.empty(13, dtype=pd.DataFrame)
G[0] = df
for i in range(0,12):
    if len(G[i]) == 1:
        G[i+1] = G[i]
    else:    
        if G[i][i].mean() >= .5:
            n = 0
        else:
            n = 1
        G[i+1] = G[i][G[i][i]==n]
    
oxy = F[12].iloc[0]@(2**np.arange(12)[::-1])
co2 = G[12].iloc[0]@(2**np.arange(12)[::-1])
life = oxy*co2

print('Answer to day3/part2: life= '+str(life)) 