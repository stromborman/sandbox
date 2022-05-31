#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 19
https://adventofcode.com/2021/day/19

Problem takes place in Z^3, input is a list whose items are lists of points in Z^3.

Context of the problem is there are unknown number of beacons (points) in Z^3.
Also in Z^3 are a collection of scanners which report back to us 
the coordinates of any beacons within 1000 (L^inf norm) of the beacon.

The issue is that the reports are in terms of (unknown) local coordinates 
centered at the scanner whose location in Z^3 is unknown.  E.g. if a scanner 
is at pt_s (unknown) in Z^3, then for some g_s (unknown) in SO(Z,3)
every beacon pt_b within range will be reported at the location g_s(pt_b - pt_s)

These local coordinates can be glued together since we are assured that if
any two scanners overlap on 12 beacons then they can be 'glued'.
We used having 3 beacons in each scanner such that the set of distances
intersected in a size >= 12 to match this criterion.
"""
import scipy.spatial.transform
import numpy as np
from collections import deque

"""
The group SO(Z,3) as matrices.
"""
so = [rot.as_matrix().astype(int) for rot in scipy.spatial.transform.Rotation.create_group("O")]
"""
By default the package provides the transformations in terms of floats.
This caused issues because exact equality is needed in the Scanner.align() method
to find the proper rotation in SO(Z,3)
"""

def load(filename):
    with open(filename) as file:
        scanners = file.read().split('\n\n')
        scanners = [item.split('\n')[1:] for item in scanners]
        scanners = [[np.array(list(map(int,item.split(',')))) for item in scanner] for scanner in scanners]
    return scanners

class Scanner:
    def __init__(self,num,beacons,rot=so[3],cent=np.array([0,0,0]),aligned = False):
        self.num = num
        self.beacons = beacons
        self.tuples = set()
        self.rot = rot
        self.cent = cent
        self.aligned = aligned
        self.dist = self.computeDists(self.beacons)
    
    def __repr__(self):
        return 'Scanner:'+str(self.num)
    
    @staticmethod
    def computeDists(beacons):
        out = {}
        for i,beacon in enumerate(beacons):
            out[i] = set()
            for beacon1 in beacons:
                out[i].add(int(np.linalg.norm(beacon1-beacon,2)**2))
        return out
    
    def checkOverlap(self,scanner1):
        pairs = []
        for key, value in self.dist.items():
            for key1, value1 in scanner1.dist.items():
                if len(value.intersection(value1)) > 11:
                    pairs.append([key, key1])
        return pairs[:3]
    
    def set_alignment(self, rot=so[3], cent = np.array([0,0,0])):
        self.rot = rot
        self.cent = cent
        self.tuples = set()
        self.aligned = True
        for beacon in self.beacons:
            self.tuples.add(tuple(rot@beacon +cent))
            
    def align(self,scanner1,pairs):
        for i, rot in enumerate(so):
            cents = []
            for pair in pairs:
                cents.append(-rot@(scanner1.beacons[pair[1]]) + self.beacons[pair[0]])
            if np.array_equal(cents[0], cents[1]) and np.array_equal(cents[1], cents[2]):
                return rot, cents[0]
    
    def propagate_align(self, scanner1, pairs): 
        rot1, cent1 = self.align(scanner1,pairs)
        total_rot = self.rot @ rot1
        total_cent = self.rot@(cent1) + self.cent
        scanner1.set_alignment(total_rot,total_cent)
        return scanner1.tuples
    
"""
For part 1: We need to find out how many beacons there are.
"""
def glue(scanners):
    scanners[0].set_alignment()
    beacons = scanners[0].tuples
    aligned = set([scanners[0]])
    notaligned = deque(range(1,len(scanners)))
    while notaligned:
        # print(notaligned)
        i = notaligned.popleft()
        work = scanners[i]
        for scanner in aligned:
            # print('checking',work,'with',scanner)
            if work.aligned == False:
                pairs = scanner.checkOverlap(work)
                if len(pairs) == 3:
                    beacons = beacons.union(scanner.propagate_align(work, pairs))
                    # print(work,'should now aligned')
        if work.aligned:
            aligned.add(work)
        else:
            notaligned.append(i)
    return beacons    

    
testdata = load('input19_t1')    
testscanners = [Scanner(i,data) for i,data in enumerate(testdata)]    
print('Passed Test for Part1:', len(glue(testscanners))==79)

realdata = load('input19')
realscanners = [Scanner(i,data) for i,data in enumerate(realdata)]
print('Answer to part1:')
print(len(glue(realscanners))) # 355



"""
For part 2: We need to find the maximum L1-distance between the scanners
"""
distances = [np.linalg.norm(item1.cent-item2.cent,1) for item1 in realscanners for item2 in realscanners]

print('Answer to part2:')
print(int(max(distances))) #10842