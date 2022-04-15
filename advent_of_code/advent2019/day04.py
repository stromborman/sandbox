#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 04
https://adventofcode.com/2019/day/4
"""

import math
import re
from itertools import combinations, groupby


def f(n,k):
    return math.comb(n+k-1,k-1)

f(7,7) # count of 2xxxxx nondecreasing

f(2,7) # strictly increasing 2xxxxx

f(2,7) # larger than 700000

f(7,5) # with form 22xxxx

f(6,4) # with form 233xxx


print('Answer to part1:', f(7,7)-f(2,7)-f(2,7)-f(7,5)-f(6,4))



div = combinations(range(14),6)
part2list = []

for x in div:
    lst = [x[i]-i+1 for i in range(6)]
    s = ''.join(map(str,lst))
    flag = any(len(list(v)) == 2 for _, v in groupby(s))
    if flag == True:
        part2list.append(x)

def part2num(lst):
    num = [lst[i]-i+1 for i in range(6)]
    s = ''.join(map(str,num))
    return int(s)
    

nums = [part2num(x) for x in part2list if 234000 < part2num(x) <700000]

print('Answer to part2:', len(nums))               