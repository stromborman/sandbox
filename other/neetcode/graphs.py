#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Graphs
"""

from collections import defaultdict, deque, Counter
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq


        
        
"""
Number of Islands:
    Given an m x n 2D binary grid grid which represents a 
    map of '1's (land) and '0's (water), return the number of islands.
"""

def numIslands(grid: List[List[str]]) -> int:
    m = len(grid)
    n = len(grid[0])

    land = {(i,j) for i in range(m) for j in range(n) if grid[i][j]=="1"}

    def nbhd(tup):
        i, j = tup
        return [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

    islands = 0

    while land:
        islands += 1
        exploring = {land.pop()}
        while exploring:
            work = exploring.pop()
            for tup in nbhd(work):
                if tup not in land:
                    continue
                exploring.add(tup)
                land.discard(tup)

    return islands
        

"""
Max Area of Island:
    You are given an m x n binary matrix grid. An island is a group of 1's.
    connected 4-directionally (horizontal or vertical.) 
    Return the maximum area of an island in grid.
"""   

def maxAreaOfIsland(grid: List[List[int]]) -> int:
     m = len(grid)
     n = len(grid[0])

     land = {(i,j) for i in range(m) for j in range(n) if grid[i][j]==1}

     def nbhd(tup):
         i, j = tup
         return [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

     best = 0
     cur = 0

     while land:
         cur = 0
         exploring = {land.pop()}
         while exploring:
             work = exploring.pop()
             cur += 1
             best = max(best,cur)
             for tup in nbhd(work):
                 if tup not in land:
                     continue
                 exploring.add(tup)
                 land.discard(tup)

     return best

"""
Pacific Atlantic Water Flow:
    There is an m x n rectangular island that borders both the Pacific Ocean and 
    Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, 
    and the Atlantic Ocean touches the island's right and bottom edges.
    
    Water can flow to neighboring cells directly north, south, east, and west 
    if the neighboring cell's height is less than or equal to the 
    current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

    Return a 2D list of grid coordinates result from where water can flow to both oceans.
"""

def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:
    m = len(heights)
    n = len(heights[0])
    
    def nbhd(tup):
        i, j = tup
        return [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
    
    def flowUp(locs):
        stack = {tup for tup in locs}
        while stack:
            loc = stack.pop()
            i0, j0 = loc
            for tup in nbhd(loc):
                i1,j1 = tup
                if 0<=i1<m and 0<=j1<n:
                    if tup not in locs:
                        if heights[i1][j1] >= heights[i0][j0]:
                            locs.add(tup)
                            stack.add(tup)
        return locs
    
    pac =  {(i, 0) for i in range(m)} | {(0, j) for j in range(n)}
    atl = {(i, n-1) for i in range(m)} | {(m-1, j) for j in range(n)}
    
    return [list(x) for x in flowUp(pac) & flowUp(atl)]


"""
Surrounded Regions:
    Given an m x n matrix board containing 'X' and 'O', capture all regions 
    that are 4-directionally surrounded by 'X'.  A region is captured by 
    flipping all 'O's into 'X's in that surrounded region.  Modify the
    board in place.
"""


def solve(board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    
    m = len(board)
    n = len(board[0])

    land = {(i,j) for i in range(m) for j in range(n) if board[i][j]=="O"}

    def nbhd(tup):
        i, j = tup
        return [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

    island = set([])

    while land:
        island = set([])
        exploring = {land.pop()}
        surronded = True
        while exploring:
            work = exploring.pop()
            island.add(work)
            i, j = work
            if surronded:
                if i in [0,m-1] or j in [0,n-1]:
                    surronded = False
            for tup in nbhd(work):
                if tup not in land:
                    continue
                exploring.add(tup)
                land.discard(tup)
        if surronded:
            for loc in island:
                i,j = loc
                board[i][j] = "X"

