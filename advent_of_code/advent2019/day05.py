#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 05
https://adventofcode.com/2019/day/5
"""
import copy

def read(filename):
    with open(filename) as file:
        nums = []
        for num in file.read().split(','):
            nums.append(int(num))
    return nums

realtape = read('input05')
test = [1002,4,3,4,33]

class Machine:
    def __init__(self, tape = [], inp = 0):
        self.tape = copy.deepcopy(tape)
        self.inp = inp
        self.pos = 0
        self.out = []
        self.flag = False
        
    def step(self):
        fullop = str(self.tape[self.pos]).zfill(5)
        op = fullop[3:]
        
        if op not in {'01','02','03','04','05','06','07','08','99'}:
            self.flag = True
            print('Error: invaid op')
        else:
            if op == '99':
                # print('99 output ', self.out)
                self.flag = True
            elif op == '03': #assuming 00103 not possible
                self.tape[self.tape[self.pos+1]] = self.inp
                self.pos = self.pos+2
            else:
                p1 = int(fullop[2])
                if p1==0:
                    x1 = self.tape[self.tape[self.pos+1]]
                else:
                    x1 = self.tape[self.pos+1]
                if op == '04':
                    self.out.append(x1)
                    self.pos = self.pos + 2
 
                if op in {'01','02','05','06','07','08'}:
                    p2 = int(fullop[1])
                    if p2==0:
                        x2 = self.tape[self.tape[self.pos+2]]
                    else:
                        x2 = self.tape[self.pos+2]
                    if op == '01':
                        self.tape[self.tape[self.pos+3]] = x1 + x2
                        self.pos = self.pos + 4
                    elif op == '02':
                        self.tape[self.tape[self.pos+3]] = x1 * x2
                        self.pos = self.pos + 4
                    elif op =='05':
                        if x1 != 0:
                            self.pos = x2
                        else:
                            self.pos = self.pos + 3
                    elif op == '06':
                        if x1 == 0:
                            self.pos = x2
                        else:
                            self.pos = self.pos + 3  
                    elif op == '07':
                        if x1 < x2:
                            self.tape[self.tape[self.pos+3]] = 1
                        else:
                            self.tape[self.tape[self.pos+3]] = 0
                        self.pos = self.pos + 4
                    elif op == '08':
                        if x1 == x2:
                            self.tape[self.tape[self.pos+3]] = 1
                        else:
                            self.tape[self.tape[self.pos+3]] = 0
                        self.pos = self.pos + 4                 
        
        return self
        

testM = Machine(test)
if testM.step().tape != [1002, 4, 3, 4, 99]:
    print('Simple example failed')
    
def run(mach,i):
    mach.flag = False
    mach.inp = i
    while mach.flag == False:
        # print(mach.pos)
        mach.step()
    return mach.out[-1]
        
realM = Machine(realtape)
            
print('Answer to part1:', run(realM, 1))


realM = Machine(realtape)
print('Answer to part1:', run(realM, 5))

