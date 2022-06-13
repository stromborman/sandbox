#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Greedy
"""

from collections import defaultdict, deque, Counter
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

"""
Hand of Straights:
    Determine if the array can be paritioned into
    arrays of length k of consecutive numbers
"""

def isNStraightHand(hand: List[int], groupSize: int) -> bool:
    n = len(hand)
    if n % groupSize != 0:
        return False

    hand = Counter(hand)
    values = list(hand.keys())
    heapq.heapify(values)
    
    while values:
        min_val = values[0]
        for card in range(min_val, min_val+groupSize):
            if card not in hand:
                return False
            hand[card] -= 1
            if hand[card] == 0:
                if card != values[0]:
                    return False
                heapq.heappop(values)
    return True




def isNStraightHandSlow(hand: List[int], groupSize: int) -> bool:
    n = len(hand)
    if n % groupSize != 0:
        return False

    runs = n//groupSize
    hand = Counter(hand)

    for i in range(runs):
        # slow because we keep recalc min from scratch
        min_value = min(hand.keys())
        for i in range(min_value, min_value+groupSize):
            if 1 < hand[i]:
                hand[i]-=1
            elif 1 == hand[i]:
                del hand[i]
            else:
                return False
    return True




"""
Merge Triplets to Form Target Triplet:
    Have a list of triplets [[a1,b1,c1],[a2,b2,c2],..]
    with operation (t1,t2): t2 = max(t1,t2) (componentwise)
    Determine if it is possible to get target triplet into the list.
"""


def mergeTriplets(triplets: List[List[int]], target: List[int]) -> bool:
    
    goal = set([])
    
    for trip in triplets:
        if trip[0] > target[0] or trip[1] > target[1] or trip[2] > target[2]:
            continue
        for i in range(3):
            if trip[i] == target[i]:
                goal.add(i)
    
    return len(goal) == 3





def mergeTripletsSlow(triplets: List[List[int]], target: List[int]) -> bool:
  
    for i, tar in enumerate(target):
        triplets = [trip for trip in triplets if trip[i] <= tar]

    correct = [[],[],[]]
    for i in [0,1,2]:
        correct[i] = [trip for trip in triplets if trip[i]==target[i]]
        if correct[i] == []:
            return False

    return True

"""
Partition Labels:
    You are given a string s. We want to split the string into as many 
    parts as possible so that no letter appears in more than one part.
    Return a list of integers representing the size of the parts (going left to right).
"""

def partitionLabels(s: str) -> List[int]:
    last = {c:i for i,c in enumerate(s)}
    ans = []
    start = end = 0
    
    for i, c in enumerate(s):
        end = max(end,last[c])
        if i == end: # iff last[c] <= i for all c so far
            ans.append(i+1-start) # make a split at <=i, with this length
            start = i+1 # new part begins
        
    return ans




def partitionLabels2(s: str) -> List[int]:
    first_last = {}
    for i,c in enumerate(s):
        if c not in first_last.keys():
            first_last[c] = [i,i]
        else:
            first_last[c][1] = i
    
    pts = set(range(len(s)+1))
    for item in first_last.values():
        start, end = item[0]+1, item[1]+1
        pts = pts - set(range(start,end))
        
    pts = sorted(list(pts))
    return [pt - pts[i] for i, pt in enumerate(pts[1:]) ]

partitionLabels("ababcbacadefegdehijhklij")





"""
Valid Parenthesis String:
    Given a string s containing only three types of characters '(', ')', '*' 
    return true if s can be made valid by picking each '*' as '' or a par.
"""

def checkValidString(s: str) -> bool:
    
    stack_par = []
    stack_star = []
    
    for i, c in enumerate(s):
        if c == '*':
            stack_star.append(i)
        if c == '(':
            stack_par.append(i)
        if c == ')':
            if stack_par:
                stack_par.pop()
            elif stack_star:
                stack_star.pop()
            else:
                return False  

    if len(stack_par) > len(stack_star):
        return False

    while stack_par:
        if stack_par.pop() > stack_star.pop():
            return False

    return True