#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Arrays & Hashing
"""

from collections import defaultdict
from typing import List
import random


"""
Group anagrams:
    Partition list of strings via anagram equivalence relation.
"""

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    eq_classes = defaultdict(list)
    for s in strs:
        eq_classes[tuple(sorted(s))].append(s)
    return eq_classes.values()


"""
Top K Frequent Elements:
    Given an integer array nums and an integer k, 
    return the k most frequent elements.
"""

def topKFrequent(nums: List[int], k: int) -> List[int]:
    d = defaultdict(int)
    for num in nums:
        d[num] += 1
    return sorted(d.keys(), key = lambda k: -d[k])[:k]


"""
Product of Array Except Self:
    Given an integer array nums, return an array answer such that answer[i] is 
    equal to the product of all the elements of nums except nums[i].
    Do no use division and make answer O(n).
"""

def productExceptSelf(nums: List[int]) -> List[int]:
    start = 1
    prefix = [1]
    for num in nums[:-1]:
        start = start*num
        prefix.append(start)
    
    end = 1
    suffix = [1]
    for num in nums[:0:-1]:
        end = end*num
        suffix.append(end)
    
    suffix = suffix[::-1]
    
    return [prefix[i]*suffix[i] for i in range(len(nums))]
        

"""
Valid Sudoku:
"""

def isValidSudoku(board: List[List[str]]) -> bool:
        
    def helper(nine):
        d = set()
        for char in nine:
            if char != '.':
                if char in d:
                    return True
                else:
                    d.add(char)
        return False
    
    for i in range(9):
        if helper(board[i]):
            return False
        if helper([board[j][i] for j in range(9)]):
            return False
    
    for a, b in [(3*i,3*j) for i in range(3) for j in range(3)]:
        if helper([board[a+i][b+j] for i in range(3) for j in range(3)]):
            return False
    
    return True

"""
Encode and Decode Strings:
    Design an algorithm to encode a list of strings to a string. 
    The encoded string is then sent over the network and is decoded back to the 
    original list of strings.
"""

# Note that this works only because we get the coded message from the start.
# If someone arbitrarly cut off some part of the start of coded, we are hosed.

def encode(strs:List[str]) -> str:
    coded = ''
    for s in strs:
        coded += 'L:'+str(len(s))+'M:'+s
    return coded

def decode(s:str) -> List[str]:
    uncoded = []
    i = 0
    while i < len(s):
        j = s[i:].index('M')
        n = int(s[i+2:i+j])
        uncoded.append(s[i+j+2:i+j+2+n])
        i = i + j + 2 + n
    return uncoded

"""
Longest Consecutive Sequence:
    Given an unsorted array of integers nums, return the length of the 
    longest consecutive elements sequence in O(n) time.
"""

def longestConsecutive(nums: List[int]) -> int: 
    matchLR = {} #keys are left endpoints, values are right end points
    matchRL = {} #keys are right endpoints, values are left end points
    seen = set()
    for num in nums:
        if num not in seen:
            seen.add(num)
            if num in matchLR.keys() and num in matchRL.keys():
                left = matchRL.pop(num)
                right = matchLR.pop(num)
            elif num in matchLR.keys():
                left = num - 1
                right = matchLR.pop(num)
            elif num in matchRL.keys():
                left = matchRL.pop(num)
                right = num+1
            else:
                left = num - 1
                right = num + 1
            matchLR.update({left:right})
            matchRL.update({right:left})
         
    return max([key-value-1 for key,value in matchRL.items()]+[0])  
