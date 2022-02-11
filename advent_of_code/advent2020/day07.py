#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 07
"""
import re
from queue import Queue

def read(filename, num=False):
    with open(filename) as file:
        rules = {}
        for line in file.read().split('\n')[:-1]:
            line = line.strip('.')
            outer, inner = line.split(' bags contain ')
            if num is True:
                inner = re.sub(r'no other', '0 no other', inner)
            inner = re.sub(r'\sbags?', '', inner).split(', ')
            if num is False:
                inner = [re.sub(r'\d\s','', item) for item in inner]
            rules[outer] = inner
    return rules

def bfs(debug=False):
    br = read('input07')
    edges = [(x[0], y) for x in br.items() for y in x[1]]

    q = Queue()
    visited = set([])
    q.put('shiny gold')
    while q.empty() == False:
        col = q.get()
        for new_col in br.keys():
            if debug: print('think', new_col)
            if (new_col, col) in edges:
                if debug: print(col, 'fits in', new_col)
                visited = visited.union(set([new_col]))
                q.put(new_col)

    return len(visited)



print('Answer to part1:', bfs())

brn = read('input07', num=True)

def count(col):
    if col == 'no other':
        return 0
    else:
        return sum( [int(nhb[0])*(1+count(nhb[2:])) for nhb in brn[col] ])

print('Answer to part2:', count('shiny gold') )
               
