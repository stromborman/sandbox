#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 03
https://adventofcode.com/2019/day/3
"""

import numpy as np
from numpy import linalg

def read(filename):
    with open(filename) as file:
        wire = []
        for string in file.read().split('\n'):
            wire.append(string.split(','))
    return wire

real = read('input03')
test1 = read('input03t1')

def wire2set(lst):
    loc = (0,0)
    wirelist = []
    for x in lst:
        if x[0] == 'R':
            vect = np.array([1,0])
        elif x[0] == 'L':
            vect = np.array([-1,0])
        elif x[0] == 'U':
            vect = np.array([0,1])
        else:
            vect = np.array([0,-1])
        
        for i in range(1,int(x[1:])+1):
            loc = tuple(np.add(loc,vect))
            wirelist.append(loc)
            
    return wirelist

def cross(lst):
    return set(wire2set(lst[0])).intersection(set(wire2set(lst[1])))
        

print('Answer to part1:', min([linalg.norm(x, ord=1) for x in cross(real)]))


print('Answer to part2:', min([wire2set(real[0]).index(x) + wire2set(real[1]).index(x) for x in cross(real)])+2)


               