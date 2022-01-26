#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 6
"""
import numpy as np


input_test = np.genfromtxt('input6_test', delimiter=',',dtype=int)
input = np.genfromtxt('input6', delimiter=',',dtype=int)

def day(a):
    b = a-1
    b[b==-1] = 6 
    c = [8]*(np.count_nonzero(a==0))
    return np.sort(np.append(b, c).astype('int'))

def run(a, n, debug = False):
    if debug == True:
        print(a)
    for i in range(n):
        a = day(a)
        if debug == True:
            print(a)
    return a

print('Answer to day6/part1: '+str(run(input, 80).shape[0]))

from itertools import groupby

def a2g(a, pad = True):
    if pad==True:
        a = np.append(a, [0,1,2,3,4,5,6,7,8])
    a = np.sort(a)
    group_a = np.array([[k,len(list(g))] for k, g in groupby(a)])
    if pad ==True:
        for i in range(9):
            group_a[i,1] = group_a[i,1]-1
    return group_a

def day_g(a):
    a0 = a[0,1]
    for i in range(8):
        a[i,1] = a[i+1,1]
    a[8,1] =  a0
    a[6,1] = a[6,1] + a0
    return a

def run_g(a,n):
    for i in range(n):
        a = day_g(a)
    return a

def test_convert(n=20):
    for i in range(n):
        if np.array_equal(run_g(a2g(input_test), n), a2g(run(input_test, n))):
            print('Sucess at n= '+str(i))
        else:
            print('Failure at n= '+str(i))        

def test_big():
    return 26984457539 == np.sum(run_g(a2g(input_test),256),axis=0)[1]

print('Answer to day6/part2: '+str(np.sum(run_g(a2g(input),256),axis=0)[1]))


# def day_g(a):
        