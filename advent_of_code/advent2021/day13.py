#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 13
"""
import pandas as pd
import numpy as np
import scipy.sparse as spa
import sys

np.set_printoptions(threshold=1000)

test = pd.read_csv('input13_test', header=None, skipfooter=2, engine='python', names=['x','y'])
test_folds = pd.read_csv('input13_test', header=None, skiprows=18, engine='python')
input = pd.read_csv('input13', header=None, skipfooter=12, engine='python', names=['x','y'])
input_folds = pd.read_csv('input13', header=None, skiprows=1004, engine='python')


mat_t = [(test['x'][i], test['y'][i]) for i in range(len(test))]

mat_i = [(input['x'][i], input['y'][i]) for i in range(len(input))]

def xfold(mat,n):
    mat_new = []
    for a in mat:
        if a[0] < n: mat_new = mat_new + [(a[0], a[1])]
        if a[0] > n: mat_new = mat_new + [(2*n - a[0], a[1])]
    return mat_new
    
def yfold(mat,n):
    mat_new = []
    for a in mat:
        if a[1] < n: mat_new = mat_new + [(a[0], a[1])]
        if a[1] > n: mat_new = mat_new + [(a[0], 2*n - a[1])]
    return mat_new

print('Answer to part1: '+ str(len(set(xfold(mat_i,655)))))

print('Answer to part2: ')

folds = pd.DataFrame(columns=['axis','on'])
folds['axis'] = input_folds[0].str.extract(r'(x|y)')
folds['on'] = input_folds[0].str.extract(r'(\d+)').astype(int)

def folder(mat, how):
    m = mat
    for i in range(len(how)):
        if how['axis'][i] == 'x':
            m = xfold(m, how['on'][i])
        else:
            m = yfold(m, how['on'][i])
    return list(set(m))

f_i = folder(mat_i, folds)
data = [1]*len(f_i)
r = [y for (x,y) in f_i]
c = [x for (x,y) in f_i]

out = spa.coo_matrix((data, (r,c)), shape = (6, 40))


a= out.toarray()

for i in range(8):
    print(a[0:6,5*i:5*i+4])
