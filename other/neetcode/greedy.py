#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Greedy
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq


        
        
"""
Maximum Subarray:
    Find the max sum of a contiguous subarray
"""

def maxSubArray(nums: List[int]) -> int:
    best = float('-inf')
    cursum = 0

    for num in nums:
        cursum += num
        best = max(best,cursum) 
        if cursum < 0:
            cursum = 0
    return best
        
    
# for nums in [[-2,1,-3,4,-1,2,1,-5,4],[1],[5,4,-1,7,8],[-5,-1,-3],[0,-3,1,1]]:
#     print(maxSubArray(nums))

"""
Jump Game:
    You are given an integer array nums, which represents the max range
    you can advance along the array.  Determine if you can reach the end.
"""

def canJump(nums: List[int]) -> bool:
    n = len(nums)
    best = 0
    for i,num in enumerate(nums):
        best = max(best,i+num)
        if best <= i < n-1:
            return False
    return True

"""
Jump Game II:
    Same set-up and assume you can reach the end.
    Return the minimum number of jumps needed.
"""

def jump(nums: List[int]) -> int:
    n = len(nums)
    next_jump = 0
    jumps = 0
    
    for i,num in enumerate(nums[:-1]):
        if i == next_jump:
            jumps += 1
            if n-1 <= i+num:
                return jumps
            else:
                choices = [i+j for j in range(num+1) if i+j < n]
                next_jump = max(choices, key=(lambda x: x+nums[x]) )
    return jumps


"""
Gas Station:
    There are n gas stations along a circular route.
    Location i has gas[i] to buy and it costs cost[i] to get to i+1.
    
    Starting with zero gas, find the unique starting location
    (if it exists) from where a complete circuit can be made.
"""


def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    if sum(gas) < sum(cost):
        return -1
    
    n = len(gas)
    delta = [gas[i]-cost[i] for i in range(n)]
    
    start = n-1
    
    tank = 0
    for i in range(n):
        tank += delta[(n-1 + i) % n]

        while tank < 0:
            if i == start:
                return -1
            start -= 1
            tank += delta[start]
        
        if i == start:
            return start
