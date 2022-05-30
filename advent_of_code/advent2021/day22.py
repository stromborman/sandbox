#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 22
https://adventofcode.com/2021/day/22

Problem takes place in the lattice Z^3
Input is a list with items of the form:
    on/off, finite 3d box
which is though of as a instruction for turning on/off
all the lattice points in the given box.
 
Solution requires determining how many lattice points are
on after completing the instructions.
"""
import re
import numpy as np
from copy import deepcopy


"""
For part 1: We are restricted from Z^3 to [-50,50]^3 so a brute force
solution of keeping track of each lattice point that gets turned on is feasiable
"""

def get_data(filename):
    with open(filename) as file:
        commands = file.read().split('\n')
        commands = [re.split(r'\sx=|,y=|,z=|\.\.', line) for line in commands]
        for item in commands:
            if item[0] == 'on':
                item[0] = True
            if item[0] == 'off':
                item[0] = False
        final = []
        for item in commands:
            item = [item[0]] + list(map(int,item[1:]))
            final.append(item)        
    return final

class LatticePoints:
    def __init__(self,bound=np.inf):
        self.elements = {}
        self.bound = bound
        
    def adjPoint(self,tuple,value):
        self.elements[tuple] = value

    def adjBlock(self, lst):
        new = [0]*6
        for i in [1,3,5]:
            new[i-1] = max(lst[i],-self.bound)
            new[i] = min(lst[i+1], self.bound)
        for x in range(new[0],new[1]+1):
            for y in range(new[2],new[3]+1):
                for z in range(new[4],new[5]+1):
                    self.adjPoint((x,y,z),lst[0])
    
    def startUp(self, filename):
        lst = get_data(filename)
        for i, item in enumerate(lst):
            self.adjBlock(item)
        return self
    
real50 = LatticePoints(bound = 50).startUp('input22')

print('Answer to part1:')
print(sum(real50.elements.values()))

"""
For part 2: The size restriction is dropped, so the brute force method runs out of memory/time.
To get around this, we need to leverage the structure of boxes (instead of dealing with each point).
The issue is that that union of two boxes is no longer a box and likewise for the complement
of a box in a box.

The solution is to use inclusion-exclusion and the fact that intersections of boxes stay boxes.

|A union B| = |A| + |B| - |A cap B|
|A union B union C| = |A|+|B|+|C| + |A cap B cap C| - |A cap B| - |A cap C| - |B cap C|
|(A union B) \ C| = |A| + |B| + |A cap B cap C| - |A cap B| - |A cap C| - |B cap C|

So at each stage we keep two lists on boxes: an On list and an Off list, where
each list only contains boxes.

For the next instruction, if it is turning on then proceed like union C, else like \ C.
"""

class Box:
    def __init__(self,bounds):
        self.bounds = bounds # [minx,miny,minz,maxx,maxy,maxz]
        self.empty = True if any([self.bounds[i] > self.bounds[i+3] for i in [0,1,2]]) else False
    
    def __repr__(self):
        if self.empty:
            return None
        else:
            return str(self.bounds)
    
    def __eq__(self,other):
        if isinstance(other, Box):
            if self.empty and other.empty:
                return True
            elif self.empty or other.empty:
                return False
            else:
                return self.bounds == other.bounds
        return False
    
    def boxIntersect(self, box):
        newbounds = []
        for i in range(3):
            newbounds.append(max(self.bounds[i],box.bounds[i]))
        for i in range(3,6):
            newbounds.append(min(self.bounds[i],box.bounds[i]))
        return Box(newbounds)
    
    def boxCount(self):
        if self.empty:
            return 0
        else:
            out = 1
            for i in range(3):
                out = out * (self.bounds[i+3]+1-self.bounds[i])
            return out
    
class InclusionExclusion:
    def __init__(self,on=[],off=[]):
        self.on = on
        self.off = off
        
    def inclExclCount(self):
        pos = sum([box.boxCount() for box in self.on])
        neg = sum([box.boxCount() for box in self.off])
        return pos - neg
    
    def turn(self,sign,newbox):
        workon = deepcopy(self.on)
        workoff = deepcopy(self.off)
        out = InclusionExclusion(workon,workoff)
        if sign:
            out.on += [newbox]
        for box in self.off:
            if not box.boxIntersect(newbox).empty:
                out.on += [box.boxIntersect(newbox)]
        for box in self.on:
            if not box.boxIntersect(newbox).empty:
                out.off += [box.boxIntersect(newbox)]
        for box in out.off:
            if box in out.on:
                out.off.remove(box)
                out.on.remove(box)
        return out
    
    def intialize(self,lst):
        working = self
        for sign, box in lst:
            working = working.turn(sign,box)
        return working
          
def get_boxes(filename):
    with open(filename) as file:
        commands = file.read().split('\n')
        commands = [re.split(r'\sx=|,y=|,z=|\.\.', line) for line in commands]
        for item in commands:
            if item[0] == 'on':
                item[0] = True
            if item[0] == 'off':
                item[0] = False
        final = []
        for item in commands:
            item = [item[0]] + list(map(int,item[1:]))
            final.append([item[0], Box([item[1],item[3],item[5],item[2],item[4],item[6]])])       
    return final    
    
real = InclusionExclusion().intialize(get_boxes('input22'))    
    
print('Answer to part2:')
print(real.inclExclCount())