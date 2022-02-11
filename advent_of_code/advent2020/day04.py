#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 04

Uses Python 3.8+ for := 
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

re1 = re.compile(r'byr:(\d{4})')
re2 = re.compile(r'iyr:(\d{4})')
re3 = re.compile(r'eyr:(\d{4})')
re4 = re.compile(r'hgt:(\d+)(in|cm)')
re5 = re.compile(r'hcl:#[0-9a-f]{6}')
re6 = re.compile(r'ecl:(amb|blu|brn|gry|grn|hzl|oth)')
re7 = re.compile(r'pid:[0-9]{9}')


def field_check(lst):
    valid = []
    for pp in lst:
        n = 0         
        if any((match := re1.fullmatch(line)) for line in pp):
            if 1920 <= int(match[1]) <= 2002:
                n += 1
        if any((match := re2.fullmatch(line)) for line in pp):
            if 2010 <= int(match[1]) <= 2020:
                n += 1
        if any((match := re3.fullmatch(line)) for line in pp):
            if 2020 <= int(match[1]) <= 2030:
                n += 1        
        if any((match := re4.fullmatch(line)) for line in pp):
            if (match[2] == 'cm') and (150 <= int(match[1]) <= 193):
                n += 1
            elif (match[2] == 'in') and (59 <= int(match[1]) <= 76):
                n += 1
        if any((match := re5.fullmatch(line)) for line in pp):
            n += 1
        if any((match := re6.fullmatch(line)) for line in pp):
            n += 1
        if any((match := re7.fullmatch(line)) for line in pp):
            n += 1
        if n == 7:
            valid.append(pp)
    return valid

print('Answer to part2:', len(field_check(ppd)))               