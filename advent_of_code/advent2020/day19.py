#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 19
https://adventofcode.com/2020/day/19
"""
import re

def read(filename):
    with open(filename) as file:
       rules, messes = file.read().split('\n\n')
       messes = messes.split('\n')
       rules = [rule.split(': ') for rule in rules.split('\n')]
       for rule in rules:
           rule[1] = rule[1].split(' | ')
           rule[1] = [branch.split(' ') for branch in rule[1]]
       rulebook = {rule[0]:rule[1] for rule in rules }
       return rulebook, messes
       
    
reald, realm = read('input19')
testd, testm = read('input19t')


def reg_rule(n='0', dct=testd):
    if dct[n]==[['a']]:
        return 'a'
    elif dct[n]==[['b']]:
        return 'b'
    else:
        return '('+'|'.join([''.join([reg_rule(num,dct) for num in subrule]) \
                             for subrule in dct[n]]) + ')'

def count(dct=testd, messes = testm):
    reg = re.compile(reg_rule('0',dct))
    matches = [bool(reg.fullmatch(mess)) for mess in messes]
    return sum(matches)


print('Answer to part1:')
print(count(reald,realm))

real2d = {}
for k,v in reald.items():
    real2d[k] = v
    
real2d['8'] = [['42'],['42','8']]
real2d['11'] = [['42','31'],['42','11','31']]

def reg_rule_loop(n='0', counter=0):
    ceiling = 30
    dct = real2d
    if dct[n]==[['a']]:
        return 'a'
    elif dct[n]==[['b']]:
        return 'b'
    elif counter == ceiling:
        return ''
    else:
        return '('+'|'.join([''.join([reg_rule_loop(num, counter+1) \
                                      for num in subrule]) \
                             for subrule in dct[n]]) + ')'

def count_loop():
    reg = re.compile(reg_rule_loop())
    matches = [bool(reg.fullmatch(mess)) for mess in realm]
    return sum(matches)

print('Answer to part2:')
print(count_loop())

