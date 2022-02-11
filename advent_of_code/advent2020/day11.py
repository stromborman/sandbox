#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 11
"""
def read(filename):
    with open(filename) as file:
        return [line.strip() for line in file.readlines()]
            
real = read('input11')
test = read('input11t')

def nbhd(x, floor=test):
    rows = len(floor)
    cols = len(floor[0])
    i = x[0]
    j=x[1]
    
    if i==0:
        di = [0,1]
    elif i==rows-1:
        di = [-1,0]
    else:
        di = [-1,0,1]
    
    if j==0:
        dj = [0,1]
    elif j==cols-1:
        dj = [-1,0]
    else:
        dj = [-1,0,1]
    
    return [[i+a,j+b] for a in di for b in dj if (a!=0 or b!=0)]

nbhd_dict = {(i,j): nbhd([i,j], real) for i in range(len(real)) for j in range(len(real[0]))}

def proc(x,floor=real):
    # print('looking at nbhds of', x)
    here = floor[x[0]][x[1]]
    nbhd_stat=''
    
    for y in nbhd_dict[(x[0],x[1])]:
        nbhd_stat= nbhd_stat + floor[y[0]][y[1]]

    
    if here=='L' and nbhd_stat.count('#') ==0:
        return '#'
    elif here=='#' and nbhd_stat.count('#') >=4:
        return 'L'
    else:
        return here

    
def time(floor=test):
    rows = len(floor)
    cols = len(floor[0])
    new_floor=[]
    for i in range(rows):
        rowi = ''
        for j in range(cols):
            rowi = rowi + proc([i,j],floor)
        new_floor.append(rowi)
    return new_floor
    
def stab(floor=test):
    keep_running = True
    floor0= floor
    while keep_running:
        if time(floor0)==floor0:
            keep_running= False
        else:
            floor0= time(floor0)
    return sum([row.count('#') for row in floor0])
    

print('Answer to part1:')
print(stab(real))


def nbhd_s(x, floor=test):
    rows = len(floor)
    cols = len(floor[0])
    i = x[0]
    j = x[1]
    
    if i==0:
        di = [0,1]
    elif i==rows-1:
        di = [-1,0]
    else:
        di = [-1,0,1]
    
    if j==0:
        dj = [0,1]
    elif j==cols-1:
        dj = [-1,0]
    else:
        dj = [-1,0,1]
    
    out= []
    for a in di:
        for b in dj:
            if (a!=0 or b!=0):
                keep_looking = True
                l=1
                while keep_looking and 0<= i+l*a < rows and 0<= j+l*b < cols:
                    # print('at', [i,j], 'looking in', [a,b], 'with', l)
                    if floor[i+l*a][j+l*b] == '.':
                        l = l+1
                    else:
                        keep_looking = False
                        out.append([i+l*a,j+l*b])
    return out



sight_nbhd = {(i,j): nbhd_s([i,j],real) for i in range(len(real)) for j in range(len(real[0]))}


def proc_s(x, floor):
    # print('looking at nbhds of', x)
    here = floor[x[0]][x[1]]
    nbhd_stat=''
    
    for y in sight_nbhd[(x[0],x[1])]:
            nbhd_stat= nbhd_stat + floor[y[0]][y[1]]
    
    if here=='L' and nbhd_stat.count('#') ==0:
        return '#'
    elif here=='#' and nbhd_stat.count('#') >=5:
        return 'L'
    else:
        return here

    
def time_s(floor):
    rows = len(floor)
    cols = len(floor[0])
    new_floor=[]
    for i in range(rows):
        rowi = ''
        for j in range(cols):
            rowi = rowi + proc_s([i,j],floor)
        new_floor.append(rowi)
    return new_floor
    
def stab_s(floor):
    keep_running = True
    floor0= floor
    while keep_running:
        if time_s(floor0)==floor0:
            keep_running= False
        else:
            floor0= time_s(floor0)
    return sum([row.count('#') for row in floor0])




print('Answer to part2:')
print(stab_s(real))



