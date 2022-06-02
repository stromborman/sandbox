#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Two Pointers
"""

from collections import defaultdict
from typing import List

"""
Two Sum- Sorted Input:
    Given a 1-indexed array of integers numbers that is sorted in non-decreasing 
    order, find two distinct indices such that their numbers add up to a specific target. 
    Your solution must use O(1) space (achieved by utilizing sorted input).
"""

def twoSum_sorted(nums: List[int], target: int) -> List[int]:
    left, right = 0, len(nums)-1
    while left < right:
        cur_sum = nums[left] + nums[right]
        if cur_sum == target:
            return [left+1,right+1]
        if cur_sum < target:
            left += 1
        else:
            right -= 1
    return []

"""
3Sum:
    Given an integer array nums, return a list without duplicates of 
    all the (unordered) triplets [nums[i], nums[j], nums[k]] 
    such that i,j,k distinct and nums[i] + nums[j] + nums[k] == 0.    
    
    This solution can be improved greatly!
"""

def threeSum(nums: List[int]) -> List[List[int]]:
       
    hash = {num:index for index,num in enumerate(nums)}    
    
    def helper(nums, start, target):
        help_ans = set()
        for j,y in enumerate(nums):
            if j > start:
                z = target - y
                if z in hash:
                    k = hash[z]
                    if k > j:
                        help_ans.add(tuple(sorted([-target, y,z])))
        return help_ans
    
    answer = set()
    for i,x in enumerate(nums):
        answer = answer.union(helper(nums, i, -x))
    return [list(item) for item in answer]


"""
Container With Most Water:
    Given input heights a list of non-negative integers,
    find the max( (j-i) * min(height[i],height[j]) )
"""
def maxArea(height: List[int]) -> int:
    left = 0
    right = len(height)-1
    best = 0
    
    while left < right:
        area = (right-left)*min(height[left],height[right])
        if area > best:
            best = area
            
        # move the min pointer giving a chance for min to increase (and hence area).
        # moving the max pointer never increases the min nor the area.
        # This provablely moves through max area via induction 
        # (take first step and immediate that max area was not in what we skipped)
        if height[right] <= height[left]:
            right -= 1
        else:
            left += 1
    
    return best


"""
Trapping Rain Water:
    Given n non-negative integers representing an elevation map where the 
    width of each bar is 1, compute how much water it can trap after raining.

    Possible to do this via a binary search where you keep splitting the
    problem into two subproblems by splitting on the max.  If a subproblem
    every has its two highest heights as end points, it stores 
    sum_i min(height[end]) - height(i)
    
    Solution below instead uses 2 pointer technique instead.
    The key observation is that the amount of water stored over i
    equals min(max(h,<i),max(h,>i)) - h[i] (provided it is nonnegative).
    Note this is equivalent to the binary search idea, since
    min(max) are just the min(end heights) for terminal splitting.
    
    With O(n) space and time we can just compute both max sequences.
    Since max(h,<i) is monotone increasing in i and max(h,>i) monotone
    decreasing, we can compute the min by just incrementing i for the smaller one
    to only use O(1) space.
"""

def trap(height: List[int]) -> int:
    left = 0
    max_l = height[left]
    
    right = len(height)-1
    max_r = height[right]
    
    total = 0
    
    while left <= right-2:            
        if max_l <= max_r:
            left += 1
            max_l = max(height[left], max_l)
            total += max(0,max_l-height[left])
        else:
            right -= 1
            max_r = max(height[right],max_r)
            total += max(0,max_r-height[right])
    
    return total