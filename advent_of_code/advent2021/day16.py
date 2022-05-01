#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 16
https://adventofcode.com/2021/day/16
"""

from dataclasses import dataclass
import math
import copy



def read(filename):
    with open(filename) as file:
        def bin_format(integer, length):
            return f'{integer:0>{length}b}'
        hexfeed = file.read().split('\n')
        binary = [bin_format(int(line, 16), 4*len(line)) for line in hexfeed]
    return binary

bin_r = read('input16')
bin_t = read('input16_test')
bin_t2 = read('input16_test2')


@dataclass
class Pack:
    version: int
    type_id: int
    length: int


@dataclass
class Oper(Pack):
    len_id: int
    subpacks: list


@dataclass
class Lit(Pack):
    value: int
    
    
    
def padcheck(bits:str) -> bool:
    for j in range(len(bits)):
        if bits[j] == '1':
            return True
    return False
    
    
def parse(bits: str, debug= False) -> Pack:
    if padcheck(bits):
    
        version = int(bits[0:3],2)
        type_id = int(bits[3:6],2)
        
        if type_id == 4:
            if debug: print('working on literal', bits[0:3], bits[3:6], bits[6:])
            s = 6
            binary = ''
            while bits[s]=='1':
                binary += bits[s+1:s+5]
                s += 5
            binary += bits[s+1:s+5]
            value = int(binary,2)
            lit = Lit(version, type_id, s+5, value)
            return lit
        
        else:
            if debug: print('working on operator', bits[0:3], bits[3:6], bits[6], bits[7:])
            len_id = bits[6]
            if len_id == '0':
                n_bits = int(bits[7:22],2)
                s = 22
                oper = Oper(version, type_id, 22+int(bits[7:22],2), len_id,[])
                if debug: print(oper,'has subpacks in next', n_bits,'bits')
                while n_bits > 0:
                    if debug: print(oper.length)
                    subpack = parse(bits[s:],debug)
                    s += subpack.length
                    n_bits -= subpack.length
                    oper.subpacks.append(subpack)
                return oper

            if len_id == '1':
                oper = Oper(version, type_id, 18, len_id, [])
                if debug: print(oper, 'with', int(bits[7:18],2), 'subpacks')
                n_sub = int(bits[7:18],2)
                s = 18
                for i in range(n_sub):
                    subpack = parse(bits[s:],debug)
                    s += subpack.length
                    oper.length += subpack.length  
                    oper.subpacks.append(subpack)    
                return oper
        

def sum_ver(pack:Pack) -> int:
    if pack.type_id == 4:
        return pack.version
    else:
        return pack.version + sum([sum_ver(subpack) for subpack in pack.subpacks])


def part1(bits) -> int:
    pack = parse(bits)
    return sum_ver(pack)


operations = {
    0: sum,
    1: math.prod,
    2: min,
    3: max,
    5: (lambda l: int(l[0] > l[1])),
    6: (lambda l: int(l[0] < l[1])),
    7: (lambda l: int(l[0] == l[1])),
}

def value(pack:Pack) -> int:
    if pack.type_id == 4:
        return pack.value
    else:
        return operations[pack.type_id]([value(subpack) for subpack in pack.subpacks]) 

def part2(bits) -> int:
    pack = parse(bits)
    return value(pack)


part1test = [6,9,14,16,12,23,31]
for i,bits in enumerate(bin_t):
    assert (part1(bits) == part1test[i]), \
        "Error for example #{} in Part 1: got {} instead of {}".format(i, part1(bits), part1test[i])
        
print('Answer to part1:', part1(bin_r[0]))


part2test= [3, 54, 7, 9, 1, 0, 0, 1]
for i,bits in enumerate(bin_t2):
    assert (part2(bits) == part2test[i]), \
        "Error for example #{} in Part 2: got {} instead of {}".format(i, part1(bits), part1test[i])
        
print('Answer to part2:', part2(bin_r[0]))
