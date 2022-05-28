#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 20
https://adventofcode.com/2021/day/20

Input consists of the following:
    1) A string of length 512 (ie 2^9) of .'s and #'s
        thought of as a function from 9bits to {.,#}
    2) A starting image in an finite grid within Z^2 
        with . or # in each spot and only . at infinity
                                                                          
Z^2 evolves in the follow manner:
    1) For each (i,j) in Z^2, view its 3x3 neighborhood as
        a 9-bit string (.=0, #=1) and use the 512-length string to look up
        a character f(i,j) in {.,#}
    2) With f(i,j) computed for all i,j.  The new image is given by f.
"""

from collections import deque
import numpy as np

class Image:
    def __init__(self, points:set, evo_rule:str,size,updates=0):
        self.points = points # the lattice points in Z^2 that are # stored at tuple
        self.evo_rule = evo_rule
        self.size = size
        self.updates = updates
        self.crop = {pt for pt in self.points if (-updates <= pt[0] < self.size+updates)
                     and (-updates <= pt[1] < self.size+updates)}
        
        
    def __repr__(self):
        minx = 0 - self.updates - 1
        maxx = self.size + self.updates + 1
        miny = 0 - self.updates - 1
        maxy = self.size + self.updates + 1
        
        string=''
        for j in range(miny,maxy):
            string += '\n'
            for i in range(minx,maxx):
                if (i,j) in self.points:
                    string += '#'
                else:
                    string += '.'
        return string[1:]
        
        
        
    def lookup(self, point):
        nbhd = list(map(tuple,np.array(point) + 
                        [np.array([-1,-1]), np.array([0,-1]), np.array([1,-1]),
                         np.array([-1,0]), np.array([0,0]), np.array([1,0]),
                         np.array([-1,1]), np.array([0,1]), np.array([1,1])]))
        bin_string = ''
        for item in nbhd:
            if item in self.points:
                bin_string+='1'
            else:
                bin_string+='0'
        return self.evo_rule[int(bin_string,2)]#, bin_string, int(bin_string,2)        
        
    def update(self):
        minx = 0 - self.updates - 1
        maxx = self.size + self.updates + 2
        miny = 0 - self.updates - 1
        maxy = self.size + self.updates + 2
        
        points_to_check = [(i,j) for i in range(minx-100,maxx+100) for j in range(miny-100,maxy+100)]
        newpoints = {point for point in points_to_check if self.lookup(point)=='#'}
        new_image = Image(newpoints,self.evo_rule,self.size,self.updates+1)
        return new_image
        
def make_image(filename):
    with open(filename) as file:
        rule, rawpoints = file.read().split('\n\n')
        rawpoints = rawpoints.split('\n')
        height = len(rawpoints)
        width = len(rawpoints[0])
        points = {(i,j) for j in range(height) for i in range(width) if rawpoints[j][i]=='#'}
    return Image(points, rule, height)
    
test = make_image('input20_test')    
real = make_image('input20')


"""
WARNING: In the test case: an array of 9 '.' gets assigned to '.', while in
the real case it gets assigned to '#'.  Then in the real case an array of 9 '#'
gets assigned to '.'  So in the real case the terms at infinity are switching between
'.' for even num of updates and '#' for odd number of updates. 
"""


"""
For part 1: We want to know how many # there are after updating twice.
"""

print('Answer to part1:', len(real.update().update().crop)) # 5437


"""
For part 2: We want to know how many # there are after updating 50 times.
"""



real50 = real
for i in range(50):
    real50 = real50.update()

print('Answer to part2:', len(real50.crop)) # 19340
    
