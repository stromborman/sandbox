#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 5
"""
import pandas as pd
import numpy as np
import scipy.sparse as spa

#input = pd.read_csv('input05_test', header=None, sep='\D+', engine='python')
input = pd.read_csv('input05', header=None, sep='\D+', engine='python')
input = input.rename(columns={0:'x0', 1:'y0', 2:'x1', 3:'y1'})
s = input.max().max()+1
    
line = input[(input.x0 == input.x1)|(input.y0 == input.y1)]
dline = input[(input.x0 != input.x1) & (input.y0 != input.y1)]

for i in range(len(line)):
    y_mi = line.iloc[i][['y0','y1']].min()
    y_ma = line.iloc[i][['y0','y1']].max()
    line.iloc[i].y0 = y_mi
    line.iloc[i].y1 = y_ma
    x_mi = line.iloc[i][['x0','x1']].min()
    x_ma = line.iloc[i][['x0','x1']].max()
    line.iloc[i].x0 = x_mi
    line.iloc[i].x1 = x_ma

def mat(i):
    S = line.iloc[i]
    rows = np.arange(S.x0, S.x1 + 1)
    cols = np.arange(S.y0, S.y1 + 1)
    prod = [(x,y) for x in rows for y in cols]
    r = [x for (x,y) in prod]
    c = [y for (x,y) in prod]
    data = [1] * len(prod)
    return spa.coo_matrix((data, (r,c)), shape = (s, s))


total_ez = spa.coo_matrix((s,s))

for i in range(len(line)):
    total_ez = total_ez + mat(i)    

print('Answer to day5/part1: '+ \
      str((total_ez/2).floor().count_nonzero())) 


for i in range(len(dline)):
    x0 = dline.iloc[i].x0
    y0 = dline.iloc[i].y0
    x1 = dline.iloc[i].x1
    y1 = dline.iloc[i].y1
    
    if x0 > x1:
        dline.iloc[i].x0 = x1
        dline.iloc[i].y0 = y1
        dline.iloc[i].x1 = x0
        dline.iloc[i].y1 = y0

def dmat(i):
    S = dline.iloc[i]
    rows = np.arange(S.x0, S.x1 + 1)
    if S.y1 >= S.y0:
        prod = [(x,S.y0 + (x-S.x0)) for x in rows]
    else:
        prod = [(x,S.y0 - (x-S.x0)) for x in rows]
    r = [x for (x,y) in prod]
    c = [y for (x,y) in prod]
    data = [1] * len(prod)
    return spa.coo_matrix((data, (r,c)), shape = (s, s))

total_d = spa.coo_matrix((s,s))

for i in range(len(dline)):
    total_d = total_d + dmat(i)
    
total = total_ez + total_d

print('Answer to day5/part2: '+ \
      str((total/2).floor().count_nonzero()))