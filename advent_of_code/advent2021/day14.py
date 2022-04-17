#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 14
https://adventofcode.com/2021/day/14
"""
import copy
from collections import Counter

def read(filename):
    with open(filename) as file:
        data = file.read().split('\n\n')
        ruleslist = [rule.split(' -> ') for rule in data[1].split('\n')]
        rules = {rule[0]:rule[1] for rule in ruleslist}
    return data[0], rules

real_input = read('input14')
test_input = read('input14_test')

class Polymer:
    def __init__(self,datarules):
        self.string = copy.deepcopy(datarules[0])
        self.rules = datarules[1]
        self.pos = 0
        self.pairs = Counter([self.string[i:i+2] for i in range(len(self.string)-1)])
        self.letters = Counter(self.string)
          
    def insert(self):
        if self.pos < len(self.string)-1:
            pair = self.string[self.pos:self.pos+2]
            if pair in self.rules.keys():
                self.string = self.string[:self.pos+1] + self.rules[pair] + \
                    self.string[self.pos+1:]
                self.pos = self.pos + 2
    
    def evo(self):
        while self.pos < len(self.string)-1:
            self.insert()
        self.pos = 0
        return self
    
    def run(self, n):
        for i in range(n):
            self.evo()
        return self
    
    def update(self,n):
        for i in range(n):
            newpairs = Counter()
            for pair in self.pairs.keys():
                n = self.pairs[pair]
                if pair in self.rules.keys():
                    X = self.rules[pair]
                    self.letters.update({X:n})
                    newpairs.update({pair[0]+X:n, X+pair[1]:n})
                else:
                    newpairs.update({pair:n})
            self.pairs = newpairs
        return self.pairs
        
        
real = Polymer(real_input)
test = Polymer(test_input)

# Part 1 was solved via brute force, by updating the entire string

part1 = Counter(real.run(10).string)        
print('Answer to part1:', part1.most_common()[0][1]-part1.most_common()[-1][1])

# Solution method for Part 1 was too slow for Part 2.  New solution method
# with Polymer.pairs, Polymer.letters, and Polymer.update() was created

real = Polymer(real_input)
real.update(40)
print('Answer to part2:', real.letters.most_common()[0][1]-real.letters.most_common()[-1][1])

