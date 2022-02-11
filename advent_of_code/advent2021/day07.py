#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 7
"""

import numpy as np

input = np.genfromtxt('input07', delimiter=',',dtype=int)
test = np.genfromtxt('input07_test', delimiter=',',dtype=int)

def fL1(n, a):
    f = 0
    for i in range(len(a)):
        f = f + np.abs(a[i]-n)
    return int(f)

# The median is the minimizer for L1-norm

print('Answer to day7/part1: ' + str(fL1(np.median(input),input)))

mean = np.mean(input)

def fMix(n, a=input):
    return int((1/2)*(np.linalg.norm(a-n)**2 + fL1(n,a)))

# The mean is the minimizer for L2-norm
# The function in question is heavily weighted as square of L2-norm

print('Answer to day7/part2: ' + str(fMix(np.floor(mean))))
