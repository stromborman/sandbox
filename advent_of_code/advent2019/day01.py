#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 01
https://adventofcode.com/2019/day/1
"""
def read(filename):
    with open(filename) as file:
        nums = []
        for num in file.read().split('\n'):
            nums.append(int(num))
    return nums

weights = read('input01')

def transform(lst):
    return [int(i/3)-2 for i in lst]

print('Answer to part1:', sum(transform(weights)) )

def intersum(n):
    fuelsum = 0
    while n > 8:
        n = int(n/3)-2
        fuelsum = fuelsum + n
    return fuelsum
        
    
print('Answer to part2:', sum([intersum(i) for i in weights]) )
               