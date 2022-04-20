#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 10
https://adventofcode.com/2021/day/10
"""
import copy
import statistics

def read(filename):
    with open(filename) as file:
        lines = file.read().split('\n')
    return lines

real_input = read('input10')
test_input = read('input10_test')

# input is a list of strings like '[({(<(())[]>[[{[]{<()<>>'


# For part 1, we need to write a parser that finds the first illegal character
# This will be achieve with the .parse method


openers = ['{','(','<','[']
closers = ['}',')','>',']']

OC = {openers[i]:closers[i] for i in range(4)}
CO = {closers[i]:openers[i] for i in range(4)}

scoreP1 = {'}':1197, ')':3,'>':25137,']':57,'':0}
scoreP2 = {'(':1,'[':2,'{':3,'<':4}

class BracString:
    def __init__(self,string):
        self.original = copy.deepcopy(string)
        self.string = string
        self.bork = ''
    
    def parse(self):
        pos = 0
        # Scan from the left until reaching a closer
        while self.bork =='' and pos < len(self.string):
            char = self.string[pos]
            if char in closers:
                # For first closer, check if previous character matches
                if pos == 0 or self.string[pos-1] != CO[char]:
                    self.bork = char
                # if there is a match, remove the pair and start scanning again
                else:
                    self.string = self.string[:pos-1]+self.string[pos+1:]
                    pos = pos - 1
            else:
                pos = pos + 1
        return self.bork

        
real = [BracString(line) for line in real_input]
test = [BracString(line) for line in test_input]

print('Answer to part1:', sum([scoreP1[string.parse()] for string in real]))

# For part 2, we are told any line that didn't bork has no other errors.
# We are asked to find the string needed to complete each of those lines
# Luckily the .parse method leaves .string with exactly when is needed
# eg '[({(<(())[]>[[{[]{<()<>>' becomes '[({([[{{'

# Reverse the order of the good part of remaining lines
good = [x.string[::-1] for x in real if x.bork=='']

def score(string):
    pts = 0
    for x in string:
        pts = pts * 5 + scoreP2[x]
    return pts

print('Answer to part2:', statistics.median([score(string) for string in good]))    