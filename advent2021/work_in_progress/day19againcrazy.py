#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 19
"""
import re
import scipy.spatial.transform

from itertools import product, combinations
from collections import defaultdict, deque
from functools import partial

import numpy as np
import numpy.linalg as la

so = scipy.spatial.transform.Rotation.create_group("O").as_matrix()

def action(n, vect):
    return (so[n]@np.array(vect)).tolist()
 

class Scanner:
    def __init__(self, id: int) -> None:
        self.id = id
        self.beacons = None
        self.diff = None
        
    def __repr__(self):
        return f'Scanner:{self.id}'
    
    def rotate(self, n) -> None:
        new_beacons = [action(n, beacon) for beacon in self.beacons]
        self.beacons = new_beacons
        
    def beacon_diff(self):
        diff = defaultdict(set)
        
        pairs = combinations(enumerate(self.beacons), 2)

        for (n, one), (m, other) in pairs:
            diff[n].add((np.array(one)-np.array(other)).tolist())
            diff[m].add((np.array(other)-np.array(one)).tolist())

        self.diff = diff       
        
        
# class Beacon:
#     def __init__(self, s_id: int, id: int, vec: np.ndarray (3,)) -> None:
#         self.s_id = s_id
#         self.id = id
#         self.vec = vec
#         self.offset = np.array([0,0,0])
        
        
        
                    
with open('input19_test') as file:
    scanners = []
    for n, group in enumerate(file.read().strip().split('\n\n')):
        new_scanner = Scanner(n)
        scanners.append(new_scanner)
        new_scanner.beacons = [tuple(map(int, line.split(','))) for line in group.split('\n')[1:]]
        


for scanner in scanners:
    scanner.beacon_diff()

def matching(scanner1, scanner2):
    translate = defaultdict(set)
    beacon_pairs = product(scanner1.beacons, scanner2.beacons)
    
    for beacon1, beacon2 in beacon_pairs:
        overlap = beacon1.dist & beacon2.dist
        if len(overlap) >= 11:
            translate[beacon1] = beacon2
    
    return translate
    
def stitch(scanners, debug=False):
    origin = scanners.popleft()
    coord = {}
    
    while scanners:
        working_scanner = scanners.popleft()
        matched_yet = False
        
        for rot in so:
            matching(origin, working_scanner.rotate(rot))
            
            if matching:
                matched_yet = True
                
        
    
    


