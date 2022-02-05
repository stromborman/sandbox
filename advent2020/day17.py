#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 17
https://adventofcode.com/2020/day/17
"""
import numpy as np
from itertools import product

def read(filename):
    with open(filename) as file:
        return [line.strip() for line in file.readlines()]
    
real = read('input17')
test = read('input17t')


def nbhd(x:tuple, dim=3): #-> list(tuple)
    d = [-1, 0, 1]
    mod = [np.array(y) for y in product(d,repeat=dim) if any([z!=0 for z in y])]
    here = np.array(x,dtype = int)
    return [tuple(z) for z in (here+mod)]

class Cube:
    def __init__(self, dim=3):
        self.elements = {}
        self.dim = dim

    def addValue(self, tuple, value):
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            value = False
        return value
    
    def countNbhd(self, x):
        return sum([self.readValue(y) for y in nbhd(x,self.dim)])
    
    def turn(self):
        turn_on = []
        consider = set(self.elements.keys())
        for z in self.elements:
            consider = consider.union(nbhd(z,self.dim))
        
        for z in list(consider):
            if self.readValue(z):
                if  2 <= self.countNbhd(z) <= 3:
                    turn_on.append(z)
            else:
                if self.countNbhd(z) == 3:
                    turn_on.append(z)
        
        self.elements = {z:True for z in turn_on}

def run(thing,dim=3):
    cube = Cube(dim)
    l = len(thing)
    
    for i,j in product(range(l),repeat=2):
        if thing[j][i]=='#':
            cube.addValue((i,j)+(0,)*(dim-2), True)
    
    for n in range(6):
       cube.turn()
    
    return len(cube.elements)

    
print('Answer to part1:')
print(run(real))




print('Answer to part2:')
print(run(real,4))
