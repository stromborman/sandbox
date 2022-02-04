#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 10
"""
from AdventUtils import read_nums_sorted as read
            
real = read('input10')
t1 = read('input10t1')
t2 = read('input10t2')

def diff(lst):
    diff = []
    for n in range(1, len(lst)):
        diff.append(lst[n] - lst[n-1])
    return diff

print('Answer to part1:')
print(diff(real).count(1) * diff(real).count(3))

# given a string of 1's of length l in diff(lst), paths(l-1) computes
# the number of ways to pick a valid subsequence for the corresponding
# section of lst
 
# counts of paths of length l on directed graph with 3 vertices
# edges: (1,1&2), (2,1&3), (3,1)

def paths(l,x=1):
    if l==-1 or l==0:
        return 1
    else:
        if x==1:
            return paths(l-1,1) + paths(l-1,2)
        elif x==2:
            return paths(l-1,1) + paths(l-1,3)
        else:
            return paths(l-1,1)

# to find the lengths of the strings of 1's we look at the differences
# between the indexes of 3's in diff.  
# The length of the string of 1's is (diff in index)-1

def runs_of_ones(lst):
    threes = [n for n, i in enumerate(diff(lst)) if i ==3]
    return [threes[0]] + [n-1 for n in diff(threes)]

# mulitply together the ways to traverse each seperate strings of 1's

def count(lst):
    c=1
    for x in runs_of_ones(lst): 
        c = c*paths(x-1)        
    return c

print('Answer to part2:')
print(count(real))
