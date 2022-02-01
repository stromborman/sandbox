#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 06
"""
from string import ascii_lowercase

def read(filename):
    with open(filename) as file:
        ans = []
        for group in file.read().split('\n\n'):
            group_ans = group.split('\n') 
            ans.append(group_ans)
        ans[-1] = ans[-1][:-1]
    return ans

ansl = read('input06')

def count(lst):
    c_any = 0
    c_all = 0
    for c in ascii_lowercase:
        if any([ c in ans for ans in lst]):
            c_any += 1
        if all([ c in ans for ans in lst]):
            c_all += 1
    return c_any, c_all        



print('Answer to part1:', sum([count(group)[0] for group in ansl]))



print('Answer to part2:', sum([count(group)[1] for group in ansl])) 
               