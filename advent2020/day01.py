#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 01
"""
from itertools import combinations

def read(filename):
    with open(filename) as file:
        nums = []
        for num in file.read().split('\n')[:-1]:
            nums.append(int(num))
    return nums

def search(lst):
    for i, j in combinations(lst, 2):
        if i+j==2020:
            n = i*j
    for i, j, k in combinations(lst, 3):
        if i+j+k == 2020:
            m = i*j*k
    return n, m

searchResults = search(read('input01')) #read and search file once
print('Answer to part1:', searchResults[0] )
print('Answer to part2:', searchResults[1] )
               