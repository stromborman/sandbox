#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 9
"""

import numpy as np
import pandas as pd

input = np.genfromtxt('input9', delimiter=1 ,dtype=int)
test = np.genfromtxt('input9_test', delimiter=1,dtype=int)

def nbhd(i,j):
    return [ [i-1,j], [i+1,j], [i,j-1], [i,j+1] ]

def check(a, i, j):
    return a[i,j] < min([ a[i0,j0] for i0,j0 in nbhd(i,j) ])

def pad(M):
    a,b = M.shape
    M = np.insert(M, b, 10, axis=1)
    M = np.insert(M, 0, 10, axis=1)
    M = np.insert(M, a, 10, axis=0)
    M = np.insert(M, 0, 10, axis=0)
    return M

def mins(M):
    m = []
    a,b = M.shape
    for i in np.arange(0,a)+1:
        for j in np.arange(0,b)+1:
            if check(pad(M), i, j):
                m.append([i-1,j-1])
    return m

def risk(M):
    r = 0
    for i, j in mins(M):
        r = r + (M[i,j] + 1)
    return r

print('Answer to day9/part1: '+str(risk(input)))

def gradflow(M, i, j, n):
    return [[a,b,n+1] for a,b in nbhd(i,j) if M[i,j] <= M[a,b] & M[a,b] < 9]

def basin(M, i, j):
    M = pad(M)
    i = i+1
    j = j+1
    b = [[i,j,0]]
    for n in range(100):
        c = 0
        for i0,j0,n0 in b:
            if n0 == n:
                b = b + gradflow(M, i0, j0, n)
                c = c + len(gradflow(M, i0, j0, n))
        if c == 0: break
    return set([ (i0-1,j0-1) for i0,j0,n0 in b])
    
def basin_df(M):
    df = pd.DataFrame(columns=['minx','miny','basin_size'])
    for i, j in mins(M):
        dict = {'minx': i, 'miny': j, 'basin_size': len(basin(M,i,j))}
        df = df.append(pd.DataFrame(dict, index=[0]), ignore_index=True)
    return df

    
    
