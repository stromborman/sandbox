# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 12
"""
import numpy as np

def read(filename):
    with open(filename) as file:
        return [[line[0], int(line[1:])] for line in file.read().split('\n')[:-1]]
    
real = read('input12')

class Ferry:
    def __init__(self, loc = [0,0], heading = 0):
        self.loc = np.array(loc)
        self.heading = heading
        
    def move(self, inst):
        act = inst[0]
        num = inst[1]
        
        if act == 'L':
            self.heading = (self.heading + num) % 360
        if act == 'R':
            self.heading = (self.heading - num) % 360
        
        if act == 'E' or (act == 'F' and self.heading == 0):
            self.loc = self.loc + [num, 0]
        if act == 'N' or (act == 'F' and self.heading == 90):
            self.loc = self.loc + [0, num]
        if act == 'W' or (act == 'F' and self.heading == 180):
            self.loc = self.loc + [-num, 0]
        if act == 'S' or (act == 'F' and self.heading == 270):
            self.loc = self.loc + [0, -num]

    def dist(self):
        return np.abs(self.loc[0]) + np.abs(self.loc[1])
    
    def course(self, lst):
        for inst in lst:
            self.move(inst)

boat = Ferry()
boat.course(real)
    
print('Answer to part1:')
print(boat.dist())

def rot(deg):
    theta = np.deg2rad(deg)
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]], dtype = int).T


class FerryW:
    def __init__(self, loc = [0,0], way = [10,1]):
        self.loc = np.array(loc)
        self.way = np.array(way)
        
    def move(self, inst):
        act = inst[0]
        num = inst[1]
        
        if act == 'L':
            self.way = rot(num)@self.way
        if act == 'R':
            self.way = rot(-num)@self.way
        
        if act == 'E':
            self.way = self.way + [num, 0]
        if act == 'N':
            self.way = self.way + [0, num]
        if act == 'W':
            self.way = self.way + [-num, 0]
        if act == 'S':
             self.way = self.way + [0, -num] 
        
        if act == 'F':
            self.loc = self.loc + num*self.way

    def dist(self):
        return np.abs(self.loc[0]) + np.abs(self.loc[1])
    
    def course(self, lst):
        for inst in lst:
            self.move(inst)
            
boatw = FerryW()
boatw.course(real)

print('Answer to part2:')
print(boatw.dist())