#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 04
"""
import re

def read(filename):
    with open(filename) as file:
        passports = [group.strip('\n') for group in file.read().split('\n\n')]
        passports = [re.split(r'\s', line) for line in passports]
    return passports

ppd = read('input04')

def quick_check(lst):
    validpp = []
    for pp in ppd:
        if len(pp) == 8:
            validpp.append(pp)
        elif len(pp) == 7 and all([not re.match(r'cid', pp[i]) for i in range(7)]):
            validpp.append(pp)
    return validpp

        

print('Answer to part1:', len(quick_check(ppd)))


re1 = r'byr:(\d{4})'
re2 = r'iyr:(\d{4})'
re3 = r'eyr:(\d{4})'
re4 = r'hgt:(\d+)(in|cm)'














print('Answer to part2:'  )               