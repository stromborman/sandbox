#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 08
"""

import copy
    
# creates inst, a list of pairs [acc/jmp/noop, int]
def read(filename):
    with open(filename) as file:
        inst = []
        for line in file.readlines():
            par = line.split(' ')
            inst.append([par[0], int(par[1])])
    return inst
            
class Prog:
    def __init__(self, inst: list) -> None:
        self.inst = inst
    
    def run(self) -> (int, bool, bool):
        no_loop = True
        in_bounds = True
        visited = {0}
        n=0
        ac=0

        while no_loop and in_bounds:
            if self.inst[n][0] == 'acc':
                ac = ac + self.inst[n][1]
                n = n + 1
            elif self.inst[n][0] == 'jmp':
                n = n + self.inst[n][1]
            else:
                n = n+1
            if n > len(self.inst):
                in_bounds = False
                break
            if n in visited:
                no_loop = False
                break
            if n == len(self.inst):
                break
            visited.add(n)
        
        return (ac, no_loop, in_bounds)

    
print('Answer to part1:')
print(Prog(read('input08')).run()[0])


# without the deepcopy flip(lst, n) output changes each time it is run 

def flip(lst, n):
    new_lst = copy.deepcopy(lst)
    # new_lst = lst
    bit = new_lst[n][0]
    if bit != 'acc':
        if bit == 'jmp':
            new_lst[n][0] = 'nop'
        else:
            new_lst[n][0] = 'jmp'
    return new_lst

def search(lst, debug=False):
    looking = True
    m = 0
    while looking:
        prog = Prog(flip(lst,m))
        if debug: print(m, prog.run())
        if prog.run()[1:] == (True, True):
            out = prog.run()[0]
            looking = False
        else:
            m = m+1
    return out
    
real = read('input08')
test = read('input08t')


print('Answer to part2:')
print(search(real))
               
