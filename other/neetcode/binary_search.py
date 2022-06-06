#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Binary Search
"""

from collections import defaultdict
from typing import List

"""
Binary Search:
    Given an array of integers nums which is sorted in ascending order, 
    and an integer target, write a function to search target in nums. 
    If target exists, then return its index. Otherwise, return -1.
"""

def search(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)-1
    
    while left <= right:
        mid = (right+left)//2
        mid_num = nums[mid]
        if mid_num==target:
            return mid
        if mid_num < target:
            left = mid + 1
        if target < mid_num:
            right = mid -1

    return -1
    
# if target not in nums, returns largest index with nums[index] < target 
def searchApprox(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)-1
    
    while left <= right and right > 0:
        mid = (right+left)//2
        mid_num = nums[mid]
        if mid_num==target:
            return nums[mid]
        if mid_num < target:
            left = mid + 1
        if target < mid_num:
            right = max(0,mid -1)

    return nums[right]

"""
Search a 2D Matrix:
    Take sorted list of length m*n and make it an mxn matrix
    in the standard way eg 1,2,3//4,5,6.
"""

def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    
    def helper(nums, target) -> int:
        left = 0
        right = len(nums)-1
        while left <= right and right > 0:
            mid = (right+left)//2
            mid_num = nums[mid]
            if mid_num==target:
                return mid
            if mid_num < target:
                left = mid + 1
            if target < mid_num:
                right = max(0,mid -1)
        return right
    
    row_index = helper([row[0] for row in matrix], target)
    
    nums = matrix[row_index]
    left = 0
    right = len(nums)-1
    
    while left <= right:
        mid = (right+left)//2
        mid_num = nums[mid]
        if mid_num==target:
            return True
        if mid_num < target:
            left = mid + 1
        if target < mid_num:
            right = mid -1

    return False
    
    
"""
Koko Eating Bananas:
    From a list of positive integers piles and an integer time h
    find the smallest positive integer k so sum_i ceil(piles[i]/k) <= h
"""    

import math

def minEatingSpeed(piles: List[int], h: int) -> int:
    bananas = sum(piles)
    left = math.ceil(bananas/h)
    if len(piles)==h:
        right = bananas
    else:
        right = math.ceil(bananas/(h-len(piles)))
    
    while left < right:
        mid = (left+right)//2
        time = sum([math.ceil(pile/mid) for pile in piles])
        if time <= h:
            right = mid
        if time > h:
            left = mid+1
            
    return left

"""
Find pivot of rotated array:
    Given rot = [i-p % n for i in range(n)], find p%n in O(log n) time.
"""

def findPivot(rot):
    n = len(rot)
    left = 0
    right = n-1
    
    while left <= right:
        if rot[left] == 0:
            return left
        
        mid = (left+right)//2
        if rot[mid] == 0:
            return mid
        
        if rot[left] > rot[mid]:
            right = mid-1
        else:
            left = mid+1
    
    return right



"""
Search in Rotated Sorted Array:
    nums[i] = ord[i-p % n] for p in range(len(ord)), where ord was ordered
    Given nums and target, return j so nums[j] = target or -1 if it doesnt exist.
"""
def searchRot(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)-1
    
    while left <= right:
        mid = (right+left)//2
       
        
        for pos in [left,mid,right]:
            if nums[pos]==target:
                return pos
        
        if nums[left] <= nums[mid] <= nums[right]:
            if target < nums[mid]:
                right = mid -1
            else:
                left = mid + 1
        
        elif nums[left] > nums[mid]:
            if nums[left] < target or target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
                
        elif nums[mid] > nums[right]:
            if nums[mid] < target or target < nums[right]:
                left = mid+1
            else:
                right = mid -1
                
    return -1
                
               
"""
Find Minimum in Rotated Sorted Array
"""    
  
  
def findMin(nums: List[int]) -> int:
    left = 0
    right = len(nums)-1
    
    while left <= right:
        mid = (left+right)//2
        
        if nums[left] <= nums[mid] <= nums[right]:
            return nums[left]
        if nums[left] > nums[mid]:
            right = mid
        elif nums[mid] > nums[right]:
            left = mid+1

    
"""
Time Based Key-Value Store:
    get(key,time) should return the value assosicated to key, whose
    timestamp is greatest amoung timestamps <= time.  
    (Return '' if this is empty, ie all timestamps for key are > time)
"""

class TimeMap:

    def __init__(self):
        self.dic=defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.dic[key].append((value,timestamp))
        
    def get(self, key: str, timestamp: int) -> str:
        series = self.dic[key]
        
        left = 0
        right = len(series)
        
        while left < right:
            mid = (left+right)//2
            
            if series[mid][1] == timestamp:
                return series[mid][0]
            
            elif series[mid][1] < timestamp:
                left = mid + 1
            else:
                right = mid
        
        if left == 0:
            return ''
        
        return series[left-1][0]
