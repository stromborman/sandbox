#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from AdventUtils import read
"""
Solution to Advent of Code 2020: Day 03
"""

def hit(x,y, debug=False):
    mnt = read('input03')
    period = len(mnt[0])
    y_max = len(mnt)
    
    count = 0
    y_loc = 0
    moves = 0
    
    while y_loc < y_max - y:
        moves = moves + 1
        y_loc = y_loc + y
        x_loc = (x*moves) % period
        if debug: print('y:', y_loc, 'x:', x_loc)
        if mnt[y_loc][x_loc] == '#':
            count = count + 1
    return count

print('Answer to part1:', hit(3,1) )
print('Answer to part2:', hit(1,1)*hit(3,1)*hit(5,1)*hit(7,1)*hit(1,2) )