# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 14
"""
import re
from itertools import product


def read(filename):
    with open(filename) as file:
        out = [line for line in file.read().split('\n')[:-1]]
        breaks = [i for i in range(len(out)) if out[i][0:4] == 'mask'] + [len(out)]
    return [out[breaks[i]:breaks[i+1]] for i in range(len(breaks)-1)]

    
real = read('input14')


class Memory:
    mask_length = 36
    
    def __init__(self):
        self.stat = {}
        self.mask = ''
    
        
    def write(self, loc:int, val:int):
        val_in_bits = bin(val)[2:].zfill(36)
        to_write = ''
        for i in range(36):
            if self.mask[i]=='X':
                to_write = to_write + val_in_bits[i]  
            else:
                to_write = to_write + self.mask[i]
        self.stat[loc] = int(to_write,2)
    
    def block(self, lst):
        self.mask = lst[0][7:]
        reg = r'mem\[(\d+)\]\s=\s(\d+)'
        for x in lst[1:]:
            parse = re.search(reg, x)
            loc = int(parse.group(1))
            val = int(parse.group(2))
            self.write(loc, val)
    
    def showme(self):
        return sum([self.stat[x] for x in self.stat.keys()])
    
def test1():
    dtm = Memory()
    for chunk in read('input14t1'):
        dtm.block(chunk)
    return dtm.showme()==165
    


docking = Memory()
for chunk in real:
    docking.block(chunk)

            
print('Answer to part1:')
print(docking.showme())

class Qmemory:
    mask_length = 36
    
    def __init__(self):
        self.stat = {}
        self.mask = ''
    
        
    def write(self, loc:int, val:int):
        loc_in_bits = bin(loc)[2:].zfill(36)
        mask_loc = ''
        for i in range(36):
            if self.mask[i]=='0':
                mask_loc = mask_loc + loc_in_bits[i]  
            else:
                mask_loc = mask_loc + self.mask[i]
        xind = [i for i,c in enumerate(mask_loc) if c=='X']       
        l = len(xind)
        if l==0:
            self.stat[int(mask_loc,2)] = val
        else:
            for op in product(['0','1'],repeat=l):
                lst = list(mask_loc)
                for i in range(l):
                    lst[xind[i]]=op[i]
                self.stat[int(''.join(lst),2)] = val
    
    def block(self, lst):
        self.mask = lst[0][7:]
        reg = r'mem\[(\d+)\]\s=\s(\d+)'
        for x in lst[1:]:
            parse = re.search(reg, x)
            loc = int(parse.group(1))
            val = int(parse.group(2))
            self.write(loc, val)
    
    def showme(self):
        return sum([self.stat[x] for x in self.stat.keys()])

def test2():
    dtqm = Qmemory()
    for chunk in read('input14t2'):
        dtqm.block(chunk)
    return dtqm.showme()==208
                
                
qdock = Qmemory()
for chunk in real:
    qdock.block(chunk)


print('Answer to part2:')
print(qdock.showme()) 