#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 02
"""


def read(filename):
    with open(filename) as file:
        return [line.strip() for line in file.readlines()]


# lines look like: 'a0-a1 letter: pw'

# check the letter appears between a0 and a1 times in pw
validpws1 = []

# check the letter appears in a0 xor a1 spot in pw
validpws2 = []

for line in read('input02'):
    rule, pw = line.split(': ')
    letter = rule.split(' ')[1]
    a0, a1 = map(int, rule.split(' ')[0].split('-'))
    
    if a0 <= pw.count(letter) <= a1:
        validpws1.append(pw)
        
    if (pw[a0-1] == letter) ^ (pw[a1-1] == letter):
        validpws2.append(pw)

print('Answer to part1:', len(validpws1) )
print('Answer to part2:', len(validpws2) )               