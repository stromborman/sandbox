#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
1-D Dynamic Programming
"""

from collections import defaultdict, deque
from typing import List, Optional
from copy import deepcopy
import heapq


        
        
"""
Climbing Stairs:
    Number of distinct ways to climb n stairs using jumps of size 1 and 2
"""

def climbStairs(n: int) -> int:
    arr = [1]*(n+1)
    for i in range(2,n+1):
        arr[i] = arr[i-1] + arr[i-2]
    return arr[-1]


"""
Min Cost Climbing Stairs:
    Minimum cost to climb to top of a set of stairs
    using jumps of size 1 and 2, where cost[i] is
    incurred when you leave the ith stair.
    Can start on 0th or 1th stair for free.
"""

def minCostClimbingStairs(cost: List[int]) -> int:
    n = len(cost)
    mincost = [0]*(n+1)
    for i in range(2, n+1):
        mincost[i] = min(cost[i-1]+mincost[i-1], cost[i-2]+mincost[i-2])
    return mincost[-1]