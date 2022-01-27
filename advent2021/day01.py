#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 1
"""

import pandas as pd

input  = pd.read_csv('input01', header=None)

df = input.rename(columns={0:'depth'})

c = 0

for i in range(1,2000):
    if df.depth[i] > df.depth[i-1]:
        c = c+1

print('Answer to day1/part1: ' + str(c))

rc = 0

for i in range(3, 2000):
    if df.depth.rolling(3).sum()[i] > df.depth.rolling(3).sum()[i-1]:
        rc = rc + 1
        
print('Answer to day1/part2: ' + str(rc))
