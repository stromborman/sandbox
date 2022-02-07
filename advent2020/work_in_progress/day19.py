#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 19
https://adventofcode.com/2020/day/19
"""
from collections import deque

def read(filename):
    with open(filename) as file:
        lst = file.readlines()
        n = lst.index('\n')
        inp = [int(item.strip().split(': ')[0]) for item in lst[:n]]
        rules = [item.strip().split(': ')[1] for item in lst[:n]]
        rules = [item.split(' | ') for item in rules]
        
        for i,x in enumerate(rules):
            if x ==['a'] or x==['b']:
                pass
            else:
                rules[i] = [list(map(int,y.split(' '))) for y in x ]
        
        rule_dct = {inp[i]:rules[i] for i in range(len(rules))}
        
        strings = [item.strip() for item in lst[n+1:]]
        
        return rule_dct, strings
    
real = read('input19')
test = read('input19t')


def run(x):
    rules= x[0]
    in_stack = deque(rules[0][0])
    queue = deque([['',in_stack]]) # a list of items [str, List[int]]
    finished = []
    
    while len(queue) > 0:
        working = queue.popleft()
        # print(working)
        string = working[0]
        stack = working[1]
        
        if len(stack) == 0:
            finished.append(string)
        
        else:
            num = stack.popleft()
            rule = rules[num]
            # print(num, rule)
            
            if rule == ['a'] or rule == ['b']:
                string = string + rule[0]
                queue.append([string,stack])
            elif len(rule) == 1:
                stack.extendleft(rule[0][::-1])
                queue.append([string,stack])
            elif len(rule) == 2:
                for item in rule:
                    newstack = stack.copy()
                    newstack.extendleft(item[::-1])
                    queue.append([string,newstack])
    
    return finished

def check(x):
    final = []
    allowed = run(x)
    for word in x[1]:
        if word in allowed:
            final.append(word)
    return final
    


print('Answer to part1:')



print('Answer to part2:')

