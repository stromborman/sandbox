#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 09
"""

from itertools import combinations
from AdventUtils import read_nums as read

real = read('input09')

def special(lst):
    searching = True
    n = 25
    while searching:
        pairs = combinations(lst[n-25:n], 2)
        if lst[n] in [i + j for i,j in pairs]:
            n = n+1
        else:
            searching = False
    return lst[n], n

ans = special(real)
    
print('Answer to part1:')
print(ans[0])

def search(lst, num):
    n = len(lst)
    target = lst[num]
    for i in range(n):
        for j in range(i,n):
            if sum(lst[i:j]) == target:
                return i, j, lst[i: j]
    
cont = search(real, 622)[2]    

print('Answer to part2:')
print(min(cont) + max(cont))
