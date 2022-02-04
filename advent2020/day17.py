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


def nbhd(x:tuple): #-> list(tuple)
    d = [-1, 0, 1]
    mod = [np.array((y0,y1,y2)) for y0,y1,y2 in product(d,repeat=3) if not (y0==y1==y2==0)]
    here = np.array(x,dtype = int)
    return [tuple(z) for z in (here+mod)]

class Cube:
    def __init__(self):
        self.elements = {}

    def addValue(self, tuple, value):
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            value = False
        return value
    
    def countNbhd(self, x):
        return sum([self.readValue(y) for y in nbhd(x)])
    
    def turn(self):
        turn_on = []
        consider = set(self.elements.keys())
        for z in self.elements:
            consider = consider.union(nbhd(z))
        
        for z in list(consider):
            if self.readValue(z):
                if  2 <= self.countNbhd(z) <= 3:
                    turn_on.append(z)
            else:
                if self.countNbhd(z) == 3:
                    turn_on.append(z)
        
        self.elements = {z:True for z in turn_on}

def run(thing):
    cube = Cube()
    l = len(thing)
    
    for i,j in product(range(l),repeat=2):
        if thing[j][i]=='#':
            cube.addValue((i,j,0), True)
    
    for n in range(6):
       cube.turn()
    
    return len(cube.elements)

    
print('Answer to part1:')
run(real)




print('Answer to part2:')

