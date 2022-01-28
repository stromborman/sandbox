#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 17
"""

import numpy as np

def y_pos(v, t):
    return (1/2)*t*(2*v+1 - t)

def traj_y(v, stop=-75):
    lst = [0]
    t = 0
    while y_pos(v, t) >= stop:
        t = t+1
        lst = lst + [y_pos(v,t)]
    return lst

# since y_pos is a deg 2 poly in t with integer roots at 0, 2v+1
# the fastest we can go is when the first neg value of y_pos = -(v+1)
# as neg as possible and still in window

print('Answer to part1: '+str(max(traj_y(73))))

def x_pos(v, t):
    if t <= v:
        return (1/2)*t*(2*v+1 - t)
    else:
        return (1/2)*v*(v+1)

test = (20, 30, -10, -5)
real = (281, 311, -74, -54)

def traj(vx, vy, stopx= 350, stopy=-100):
    lst = [(0,0)]
    t = 0
    while y_pos(vy, t) >= stopy and x_pos(vx, t) <= stopx:
        t = t+1
        lst = lst + [(x_pos(vx, t), y_pos(vy, t))]
    return lst

def hit_check(vx, vy, tar=test):
    hit = False
    for x, y in traj(vx, vy):
        if x in np.arange(tar[0], tar[1]+1) and y in np.arange(tar[2], tar[3]+1):
            hit = True
            return hit
    return hit

# Complete brute force
# Min vx = np.sqrt(2*min_x(tar)) - 1
# Max vx = max_x(tar)
# Min vy = min_y(tar)
# Max vy = -min_y(tar)
# so just brute force grid search

def dumb_search(tar=test):
    lst = []
    min_vx = 1 #int(np.sqrt(2*tar[0]) - 1)
    max_vx = tar[1]+1
    min_vy = tar[2]
    max_vy = -tar[2]+1
    for vx in np.arange(min_vx, max_vx):
        for vy in np.arange(min_vy, max_vy):
            # print('checking ', vx, vy)
            if hit_check(vx, vy, tar) is True:
                # print(hit_check(vx, vy, tar))
                lst = lst + [(vx,vy)]
    return lst

def hit_count(tar=test):
    return len(dumb_search(tar))

print('Answer to part1: '+str(hit_count(real)) )    