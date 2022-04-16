#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 02
https://adventofcode.com/2019/day/2
"""

from itertools import combinations
import copy

def read(filename):
    with open(filename) as file:
        nums = []
        for num in file.read().split(','):
            nums.append(int(num))
    return nums

tape = read('input02')

test = [1,9,10,3,2,3,11,0,99,30,40,50]

def run(lst0, n, m):
    lst = copy.deepcopy(lst0)
    lst[1] = n
    lst[2] = m
    pos = 0
    keeprunning = True
    while keeprunning == True:
        if lst[pos] == 1:
            lst[lst[pos+3]] = lst[lst[pos+1]] + lst[lst[pos+2]]
            pos = pos + 4
        elif lst[pos] == 2:
            lst[lst[pos+3]] = lst[lst[pos+1]] * lst[lst[pos+2]]
            pos = pos + 4
        else:
            keeprunning = False
    return lst[0]

print('Answer to part1:', run(tape,12,2))

def brute(lst, n):
    for i,j in combinations(range(n), 2):
        if run(lst,i,j)==19690720:
            return i, j
    
print('Answer to part2:', brute(tape,100))


               