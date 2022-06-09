#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Backtracking
"""

from collections import defaultdict, deque, Counter
from typing import List, Optional
from copy import deepcopy
import heapq


"""
Subsets
"""

def subsets(nums: List[int]) -> List[List[int]]:
    if len(nums) == 1:
        return [[],nums]
    
    return subsets(nums[:-1]) + [lst + [nums[-1]] for lst in subsets(nums[:-1])] 


"""
Combination Sums:
    Can use unlimited number of copies of numbers for list to create sum.
    Need to determine all unique ways to do this (order of + does not matter).
"""

def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    candidates = sorted(candidates)
    def subproblem(lst,tar):

        if lst == []:
            return []
        if lst:
            if min(lst) > tar:
                return []
            maxnum = max(lst)
            subsol = subproblem(lst[:-1],tar) 
            if maxnum > tar:
                return subsol
            if maxnum == tar:
                return [[maxnum]] + subsol
            else:
                return [item + [maxnum] for item in subproblem(lst,tar-maxnum)] + subsol
        
    return subproblem(candidates,target)

"""
Permutations
"""

def permute(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    if n == 1:
        return [nums]
    return [item[:i] + [nums[-1]] + item[i:] for item in permute(nums[:-1]) for i in range(n)]


"""
Subsets II:
    Given an integer array nums that may contain duplicates, 
    return all possible subsets.
    The solution set must not contain duplicate subsets.
"""

def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    ans = []
    temp = []
    nums.sort()
    
    def backtrack(start):
        print('temp',temp,'start',start)
        print('ans',ans)
        ans.append(temp.copy())
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            temp.append(nums[i])
            backtrack(i + 1)
            temp.pop()
            
    backtrack(0)
    return ans



def subsetsWithDupNaive(nums: List[int]) -> List[List[int]]:
    nums = sorted(nums)    
    subsets = []
    subsets.append([])
    
    for currentnumber in nums:
        n = len(subsets)
        for i in range(n):
            lst = list(subsets[i])
            lst.append(currentnumber)
            
            if lst not in subsets:
                subsets.append(lst)
    return subsets



"""
Combination Sum II:
    Given a collection of candidate numbers (candidates) and a target 
    number (target), find all unique combinations in 
    candidates where the candidate numbers sum to target.

    Each index in candidates may only be used ONCE in the combination.
"""

def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    
    def backtrack(comb,tar,index,counter,answers):
        if tar == 0:
            answers.append(list(comb))
            return
        
        if tar<0:
            return
        
        for i, (num, count) in enumerate(counter):
            if count <= 0 or i < index:
                continue
            
            comb.append(num)
            counter[i] = (num,count-1)
            
            backtrack(comb,tar-num,i,counter,answers)
            
            # When we get here, we have done all we could with that choice of number, so undo it
            counter[i] = (num, count)
            comb.pop()
        return answers
    

    counter = [(num,count) for num,count in Counter(candidates).items()]
    
    return backtrack([],target,0,counter,[])
        



