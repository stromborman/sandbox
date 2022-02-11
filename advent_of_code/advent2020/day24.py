#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 24
https://adventofcode.com/2020/day/24
"""

import re
from collections import Counter

import numpy as np

# input is a sequence of words in the group whose Caley graph is
# a regular tiling of the plane by triangles (dual is regular hexagrams)
# group generators are e, ne, nw, w, sw, se (compass directions)

def read(filename):
    with open(filename) as file:
        words = [re.sub(r'([ew])',r'\1,',word)[:-1].split(',') for word in file.read().split('\n')]
        return words
    
real = read('input24')
test = read('input24t')


# for part 1 we need to count occurances of unique elements mod 2
# that is #{ g in G: (# of g in list) = 1 mod 2 }

# instead of using group relations to reduce words, we will identify
# group elements by their coordinates in the plane with the identity at origin

th = np.pi/3
gens = ['e', 'ne', 'nw', 'w', 'sw', 'se']

def gen_to_corrd(gen): # group gen -> coords in plane
    n = gens.index(gen)
    return np.array([np.cos(n*th), np.sin(n*th)])

def word_to_coord(word): # group word -> final coord in plane
    coord = np.array([0,0])
    for char in word:
        coord = coord + gen_to_corrd(char)
    # use 3 for rounding since 4th decimal point of sqrt(3) and sqrt(3)/2 is 0
    return tuple(np.round(coord,decimals=3))

def count_black(lst):
    # a dict where keys are coords and values are count of occurences
    elements = Counter([word_to_coord(word) for word in lst])
    # number of coordinates that appear an odd number of times
    return sum([v % 2 for v in elements.values()])

print('Answer to part1:')
print(count_black(real))

# for part 2, each day the tiles flip according to their neighbors' statuses
# need to calculate the 100th day

def nbhd(x:tuple): #-> list(tuple)
    here = np.array(x)
    return [tuple(np.round(here + gen_to_corrd(gen),3)) for gen in gens]

class Floor:
    def __init__(self, elements):
        # a dict serving as a sparse 'matrix' of tiles
        # self.elements(tile) == True means tile is black
        # False or tile not in keys means tile is white
        self.elements = {item: True for item in elements}
        
    def addValue(self, tuple, value):
        self.elements[tuple] = value # True is black, False is white

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            value = False # tiles not in self.elements are considered white
        return value
    
    def black_count(self):
        return sum([v for k,v in self.elements.items()])
    
    def countNbhd(self, x):
        return sum([self.readValue(y) for y in nbhd(x)])
    
    # runs the rules for one day
    def turn(self):
        turn_on = []
        consider = set(self.elements.keys())
        for z in self.elements:
            consider = consider.union(nbhd(z))
        
        for z in list(consider):
            if self.readValue(z):
                # black tiles turn white unless they have 1 or 2 black nbhds
                if  1 <= self.countNbhd(z) <= 2:
                    turn_on.append(z)
            else:
                # white tiles with 2 black neighbors turn black
                if self.countNbhd(z) == 2:
                    turn_on.append(z)
        
        # tiles that are black after this round
        # could be made more efficent by not readding tiles that previously were black 
        self.elements = {z:True for z in turn_on}
        
    def run(self,n):
        for i in range(n):
            self.turn()
        return self
        

    
def list_to_floor(lst):
    # a dict where keys are coords and values are count of occurences
    elements = Counter([word_to_coord(word) for word in lst])
    black = [k for k,v in elements.items() if v%2 == 1]
    return black


print('Answer to part2:')
print(Floor(list_to_floor(real)).run(100).black_count())