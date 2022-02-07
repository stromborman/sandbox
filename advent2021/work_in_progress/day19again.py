#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 19
"""
import re
import scipy.spatial.transform

from itertools import product, combinations
from collections import defaultdict
from functools import partial

import numpy as np
import numpy.linalg as la

so = scipy.spatial.transform.Rotation.create_group("O").as_matrix() 

class Scanner:
    def __init__(self, id: int) -> None:
        self.id = id
        self.beacons = None
        self.center = None
        self.dist = None
        
    def __repr__(self):
        return f'Scanner:{self.id}'
    
    def rotate(self, n: int) -> None:
        new_beacons = [so[n]@vec for vec in self.beacons]
        self.beacons = new_beacons
        
    def beacon_diff(self):
        l = len(self.beacons)
        center = np.empty((l,l), dtype= np.ndarray)
        dist = np.empty(l, dtype= set)
        for i in range(0, l):
            for j in range(0, l):
                center[i,j] = self.beacons[j] - self.beacons[i]
            dist[i] = set([int(la.norm(center[i,j])**2) for j in range(l)])
        self.center = center
        self.dist = dist
        
class Beacon:
    def __init__(self, id: int, vec: np.ndarry (3,)) -> None:
        self.id = id
        self.vec = vec    
        
        
        
                    
with open('input19_test') as file:
    scanners = []
    for n, group in enumerate(file.read().strip().split('\n\n')):
        new_scanner = Scanner(n)
        scanners.append(new_scanner)
        new_scanner.beacons = [np.array(tuple(map(int, line.split(',')))) for line in group.split('\n')[1:]]

for scanner in scanners:
    scanner.beacon_diff()

def matching(scanner1, scanner2):
    overlaps = defaultdict(set)
    beacon_pairs = product(scanner1.beacons, scanner2.beacons)
    
    


