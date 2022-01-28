#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 19
"""

import numpy as np
import numpy.linalg as la
import pandas as pd


with open('input19_test') as file:
    lst = [1]
    n=0
    for num, line in enumerate(file, 1):
        n = n+1
        if line == '\n':
            lst = lst + [num+1]
    lst = lst + [n+2]
    
scan = []

for i in range(len(lst)-1):
     scan = scan + [pd.read_csv('input19_test', header=None, skiprows=lst[i], nrows=(lst[i+1]-lst[i])-2) ]
     
# pd.read_csv('input19_test', header=None, skiprows=lst[0], nrows=(lst[1]-lst[0])-2)      
# pd.read_csv('input19_test', header=None, skiprows=lst[1], nrows=(lst[2]-lst[1])-2)
# pd.read_csv('input19_test', header=None, skiprows=lst[2], nrows=(lst[3]-lst[2])-2)

for x in range(len(scan)):
    scan[x] = scan[x].to_numpy()

R1 = np.array([[-1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])

R2 = np.array([[1, 0, 0],
       [0, -1, 0],
       [0, 0, 1]])

R3 = np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, -1]])

P = np.array([[0, 0, 1],
       [1, 0, 0],
       [0, 1, 0]])

def affine(m, t, v):
    return m@v + t

def dist(scn):
    l = len(scn)
    mat = np.zeros((l,l), dtype=int)
    for i in range(0, l):
        for j in range(0, l):
            mat[i,j] = la.norm(scn[i] - scn[j], ord=1)
    return mat

def dist_list(scn):
    l = len(scn)
    lst = []
    for i in range(0, l):
        for j in range(0, i):
            lst = lst+ [int(la.norm(scn[i] - scn[j], ord=1))]
    return lst

def compare(i,j):
    A = dist(scan[i])
    B = dist(scan[j])
    lst = []
    for i in range(len(A)):
        for j in range(len(B)):
            if len(set(A[i]).intersection(set(B[j]))) >= 10:
                lst = lst + [(i,j)]
    return lst

expon = [(x,y,z,r) for x in range(2) for y in range(2) for z in range(2) for r in range(3)]
direc = [ la.matrix_power(R1,x)@la.matrix_power(R2,y)@la.matrix_power(R3,z)@la.matrix_power(P,r) for\
         x,y,z,r in expon]

def align(i,j):
    for M in direc:
        v = scan[0][0] - M@scan[1][3]
        if all( (v == scan[0][i] - M@scan[1][j]).all() for i,j in compare(0,1)):
            return M, v
    
    




print('Answer to part1: ' )
print('Answer to part2 ' )