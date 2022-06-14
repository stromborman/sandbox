#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Math & Geometry
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq
import random
import math


        
        
"""
Happy Number:
    f: N -> N by f(x) = sum d(x,i)^2 where d(x,i) is the ith digit of x
    For given n, determine if n flows to 1 or not.
    There are cycles 20 -> 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20
    f(x) < 100*log(x), so orbits are bounded, so must be 
    eventually periodic (or stationary) 
"""

def isHappy(n: int) -> bool:
    def f(n):
        return sum([x**2 for x in map(int,list(str(n)))])

    out = defaultdict(int)
    
    while True:
        n = f(n)
        if n == 1:
            return True
        if out[n] == 1:
            return False
        out[n]+=1

"""
Plus One:
    Given a list of int in [0,9], thought of as digits of a number x.
    Return a list of int representing x+1.
"""

def plusOne(digits: List[int]) -> List[int]:        
    num = int(''.join([str(x) for x in digits]))+1
    return [int(c) for c in str(num)]


"""
Rotate Image:
    You are given an n x n 2D matrix representing an image, 
    rotate the image by 90 degrees (clockwise).
"""

def rotate(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n = len(matrix)
    
    matrix1 = [[matrix[n-1 - j][i] for j in range(n)] for i in range(n)]
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = matrix1[i][j]


"""
Set Matrix Zeroes:
    Given an m x n integer matrix matrix, if an element is 0, 
    set its entire row and column to 0's.
"""

def setZeroes(matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    m = len(matrix)
    n = len(matrix[0])
    
    rows = [i for i, row in enumerate(matrix) if 0 in row]
    columns = [j for j in range(n) if 0 in [matrix[i][j] for i in range(m)]]
    
    for i in rows:
        matrix[i] = [0]*n
    for j in columns:
        for i in range(m):
            matrix[i][j] = 0
            
        
"""
Spiral Matrix:
    Given an m x n matrix, return all elements of the matrix in spiral order.
"""


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    def tran(mat):
        m = len(mat)
        n = len(mat[0])
        return [[mat[i][j] for i in range(m)] for j in range(n)] # tran[j] is the j-th column of matrix
    
    def helper(mat, lst):
        mrow = len(mat)
        ncol = len(mat[0])
        trans = tran(mat)

        if mrow == 1:
            return lst + mat[0]

        if ncol == 1:
            return lst + tran(mat)[0]

        if mrow == 2:
            return lst + mat[0] + mat[1][::-1]

        if ncol == 2:
            return lst + mat[0] + trans[1][1:] + trans[0][1:][::-1]

        lst = lst + mat[0][:-1] + trans[-1][:-1] + mat[-1][1:][::-1] + trans[0][1:][::-1]
        mat = [mat[i][1:-1] for i in range(1,mrow-1)]

        return helper(mat, lst)

    return helper(matrix, [])
        

spiralOrder([[1,2,3],[4,5,6],[7,8,9]])

mat = [[1,2,3],[4,5,6],[7,8,9]]


"""
Detect Squares:
    You are given a stream of points on the X-Y plane.
    Adds new points from the stream into a data structure. 
    Duplicate points are allowed and should be treated as different points.
    Given a query point, counts the number of ways to choose three points 
    from the data structure such that the three points and the 
    query point form an axis-aligned square with positive area.
"""

class DetectSquares:

    def __init__(self):
        self.points = defaultdict(int)
        self.xvalues = defaultdict(set) #for key k, gives all points on x = k 
        self.yvalues = defaultdict(set) #for key k, gives all points on y = k
        
    def add(self, point: List[int]) -> None:
        self.points[tuple(point)] += 1
        self.xvalues[point[0]].add(tuple(point))
        self.yvalues[point[1]].add(tuple(point))
        
    def count(self, point: List[int]) -> int:
        x0, y0 = point
        p0 = (x0,y0)
        ans = 0
        
        for p1 in self.xvalues[x0]:
            if p1 == p0:
                continue
            height = abs(p1[1]-p0[1])
            
            for p2 in self.yvalues[y0]:
                if p2 == p0:
                    continue
                
                if abs(p2[0]-p0[0]) != height:
                    continue
            
                p3 = (p2[0],p1[1])
                if p3 in self.points.keys():
                    ans += self.points[p1]*self.points[p2]*self.points[p3]
        
        return ans