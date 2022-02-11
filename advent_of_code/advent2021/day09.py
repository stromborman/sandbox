#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 9
"""

import numpy as np
from queue import Queue
import pandas as pd

input = np.genfromtxt('input09', delimiter=1 ,dtype=int)
test = np.genfromtxt('input09_test', delimiter=1,dtype=int)

def nbhd(i,j):
    return [ (i-1,j), (i+1,j), (i,j-1), (i,j+1) ]

def nbhd_full(i,j):
    return nbhd(i,j) + [ (i,j) ] 

def check(a, i, j):
    return a[i,j] < min([ a[i0,j0] for i0,j0 in nbhd(i,j) ])

def pad(M):
    a,b = M.shape
    M = np.insert(M, b, 10, axis=1)
    M = np.insert(M, 0, 10, axis=1)
    M = np.insert(M, a, 10, axis=0)
    M = np.insert(M, 0, 10, axis=0)
    return M

def minpts(M, prepad=True, padout=False):
    m = []
    a,b = M.shape
    if prepad == False:
        for i in np.arange(0,a)+1:
            for j in np.arange(0,b)+1:
                if check(pad(M), i, j):
                    if padout == False:
                        m.append([i-1,j-1])
                    else:
                        m.append([i,j])
    else:
        for i in np.arange(1,a-1):
            for j in np.arange(1,b-1):
                if check(M, i, j):
                    m.append([i,j])
    return m

def risk(M):
    r = 0
    for i, j in minpts(M, prepad=False):
        r = r + (M[i,j] + 1)
    return r

print('Answer to day9/part1: '+str(risk(input)))

def spreader(M, i0, j0, l=15):
    q = Queue()
    q.put((i0,j0))
    while q.empty() == False:
        i, j = q.get()
        for x in nbhd(i,j):
            if M[x[0],x[1]] < 9:
                q.put((x[0],x[1]))
                M[x[0],x[1]] = l
    return M



def bp(M):
    N = pad(M)
    for n, x in enumerate(minpts(pad(M))):
        spreader(N, x[0], x[1], 11+n)
    return N

def bs(M):
    unique, counts = np.unique(bp(M), return_counts=True)        
    df = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['basin_id', 'size'])
    df = df.drop([0,1])
    df['basin_id'] = df['basin_id'] - 11
    df = df.set_index('basin_id')
    df2= pd.DataFrame(minpts(M, prepad=False), columns=(['row','col']))
    # for x in minpts(M, prepad=False):
    #     df['minpt'] = (x[0], x[1])
    return df.join(df2).sort_values(['size'], ascending = False)

# print(bs(input).head())

def ans(M):
    ans = 1
    for i in range(3):
        ans = ans * bs(M).iloc[i,0]
    return ans
    

print('Answer to day9/part2: '+str(ans(input)))