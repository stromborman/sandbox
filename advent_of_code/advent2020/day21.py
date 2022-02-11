#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 21
https://adventofcode.com/2020/day/21


"""

import re
import functools

def read(filename):
    with open(filename) as file:
        out = [line.strip().split(' (contains ') for line in file.readlines()]
        for item in out:
            item[0] = set(item[0].split(' '))
            item[1] = set(item[1][:-1].split(', '))
        return out

test = read('input21t')
real = read('input21')

def universe(lst):
    return [functools.reduce(set.union, [item[i] for item in lst]) for i in [0,1]]

    
def deduce(lst):
    all_aller = universe(lst)[1]
    dct = {}
    for aller in all_aller:
        dct[aller] = functools.reduce(set.intersection, [item[0] for item in lst if aller in item[1]])
    
    working = True
    while working:
        working = False
        for k, v in dct.items():
            if len(v) == 1:
                for k1, v1 in dct.items():
                    if k1 != k:
                        if v <= v1:
                            working = True
                        dct[k1] = v1 - v
                        
    return dct
    
def count(lst):
    bad_ingred = functools.reduce(set.union, deduce(lst).values())
    return sum([len(item[0]-bad_ingred) for item in lst])
    
print('Answer to part1:')
print(count(real))


def danger(lst):
    foo = sorted(deduce(lst).items())
    out = ''
    for k, v in foo:
        out = out +','+v.pop()
    return out[1:]

print('Answer to part2:')
print(danger(real))
