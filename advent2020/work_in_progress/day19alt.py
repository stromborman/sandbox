#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 19
https://adventofcode.com/2020/day/19
"""
from collections import deque
import re


def read(filename):
    with open(filename) as file:
        lst = file.readlines()
        n = lst.index('\n')
        inp = [item.strip().split(': ')[0] for item in lst[:n]]
        rules = [item.strip().split(': ')[1] for item in lst[:n]]
        
        rule_dct = {inp[i]:rules[i] for i in range(len(rules))}
        strings = [item.strip() for item in lst[n+1:]]
        
        return rule_dct, strings
    
real = read('input19')
test = read('input19t')


rr = read('input19')[0]
newdct = {}
keep = True
n=0

while keep is True:
    n = n+1
    special = [(k,v) for k,v in rr.items() if bool(re.fullmatch(r'\D+',v))]
    for sk,sv in special:
        del rr[sk]
        newdct[sk] = sv
        print(sk,sv)
        if bool(re.search(r'\|',sv)):
            for k,v in rr.items():
                rr[k] = v.replace(sk,'('+sv+')')
        else:
            for k,v in rr.items():
                rr[k] = v.replace(sk,sv)
    if n == 100: keep = False

def run2():
    rules= rr
    in_stack = deque(rules['0'])
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

print('Answer to part1:')



print('Answer to part2:')