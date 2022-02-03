# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 13
"""
import re
from decimal import Decimal


def read(filename):
    with open(filename) as file:
        raw = file.read().split('\n')
        time = int(raw[0])
        buses = [int(x) for x in re.findall(r'\d+', raw[1])]
        rawb = raw[1].split(',')
        loc = [i for i, e in enumerate(rawb) if e != 'x']
    return [time, buses, loc]
        
real = read('input13')
test = read('input13t')

def first(sched):
    arr = [((-sched[0] % x), x) for x in sched[1]]
    arr.sort(key= lambda y: y[0])
    get = arr[0]
    return get[0]*get[1]

print('Answer to part1:')
print(first(real))

# using Decimal to avoid python big number issues

def crt(vals, mods):
    N = 1
    for n in mods:
        N = Decimal(N)*Decimal(n)
    NI = [Decimal(N)/Decimal(n) for n in mods]
    M = []
    for n in mods:
        for l in range(1,n):
            if int(l*N/n) % n == 1:
                M.append(l)
                break
    ans = 0
    for i in range(len(vals)):
        ans = ans + vals[i]*NI[i]*M[i]
    x = -int(ans) % int(N)
    return x

def rdt2(filename):
    with open(filename) as file:
        raw = file.read().split('\n')
        ans = int(raw[1])
        buses = [int(x) for x in re.findall(r'\d+', raw[0])]
        rawb = raw[0].split(',')
        loc = [i for i, e in enumerate(rawb) if e != 'x']
    return [ans, buses, loc]


def check():
    fnames = ['input13t' + str(i) for i in range(6)]
    tdata = [rdt2(name) for name in fnames] 
    for data in tdata:
        print(crt(data[2], data[1]), data[0], crt(data[2], data[1])==data[0])

print('Answer to part2:')
print(crt(real[2],real[1]))
