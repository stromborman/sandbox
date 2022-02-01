#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 05
"""
def read(filename):
    with open(filename) as file:
        bpass = [[line[0:7],line[7:10]] for line in file.readlines()]
    return bpass

bp = read('input05')

def convert(lst):
    bpn = []
    for item in lst:
        row = 0
        for i in range(7):
            if item[0][i] == 'B':
                row = row + 2**(6-i)
        col = 0        
        for i in range(3):
            if item[1][i] == 'R':
                col = col + 2**(2-i)
        bpn.append([row, col])
    return bpn

def seatid(lst):
    id_lst = []
    for item in convert(lst):
        id_lst.append(item[0]*8 + item[1])
    return id_lst


print('Answer to part1:', max(seatid(bp)))


def find():
    lst = sorted(seatid(bp))
    found = False
    n = 0
    while found is False:
        if lst[n] + 2 == lst[n+1]:
            found = True
            seat = lst[n] + 1
        else: n = n+1
    return seat
            
print('Answer to part2:', find())               