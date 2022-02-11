#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 18
https://adventofcode.com/2020/day/18

Order of operations for + and * are now done strictly left to right,
e.g. 2 + 3 * 5 + 4 = 5 * 5 + 4 = 25 + 4 = 29 

Parentheses guide order of operations as is standard,
e.g. 2 + ((3 * 5) * 2) = 2 + (15 * 2) = 2 + 30 = 32

Part 1 asks for an implimentation of these rules.
Part 2 asks for + to take precedence over *.
"""

import re

def read(filename):
    with open(filename) as file:
        return [line.strip() for line in file.readlines()]
    

test = read('input18t')
tans1 = [71, 51, 26, 437, 12240, 13632]
tans2 = [231, 51, 46, 1445, 669060, 23340]

real = read('input18')


reg_op = re.compile(r'(\d+)\s(\+|\*)\s(\d+)')

def oper(string): # 'num1 + num2' or 'num1 * num2'
    match = reg_op.match(string)
    if match[2] == '+':
        return str(int(match[1]) + int(match[3]))
    if match[2] == '*':
        return str(int(match[1]) * int(match[3]))

def noparen(string): # recursively works through string without paren
    match = reg_op.match(string)
    if not match:
        return string
    else:
        return noparen(oper(match[0])+string[match.end():])
    
reg_par = re.compile(r'\([^\(]*?\)')

def resolve(string):
    match = reg_par.search(string)
    if not match:
        return noparen(string)
    else:
        return resolve(string[:match.start()]+noparen(match[0][1:-1])+string[match.end():])


# tans1 == [int(resolve(item)) for item in test]


print('Answer to part1:')
print(sum([int(resolve(item)) for item in real]))


reg_add = re.compile(r'(\d+)\s\+\s(\d+)')

def addfirst(string): # recursively works through string without paren
    match = reg_add.search(string) # resolve all the additions first
    if not match: 
        return noparen(string) # once all additions are gone resolve multi
    else:
        return addfirst(string[:match.start()]+oper(match[0])+string[match.end():])

def resolve2(string):
    match = reg_par.search(string)
    if not match:
        return addfirst(string)
    else:
        return resolve2(string[:match.start()]+addfirst(match[0][1:-1])+string[match.end():])

tans2 == [int(resolve2(item)) for item in test]

print('Answer to part2:')
print(sum([int(resolve2(item)) for item in real])) 

