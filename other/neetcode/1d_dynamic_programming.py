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

"""
House Robber:
    For an array nums >= return the max sum where adjacent entries
    are not used.
"""

def rob(nums: List[int]) -> int:
    # money[i] wiil be the answer for the subproblem nums[:i+1]
    money = {i:0 for i in range(-2,len(nums))}
    for i, num in enumerate(nums):
        money[i] = max(money[i-2]+nums[i], money[i-1])
    return money[len(nums)-1]


"""
House Robber II:
    Same problem but now nums[0] and nums[-1] are considered adjacent    
"""

def rob2(nums: List[int]) -> int:
    n = len(nums)
    money = [(0,0)]*n #first entry uses nums[0], second does not
    
    for i in range(n):
        if i == 0:
            money[0] = (nums[0], 0)
        elif i == 1:
            money[1] = (nums[0],nums[1])
        elif i < n - 1:
            money[i] = (max(money[i-2][0]+nums[i], money[i-1][0]), max(money[i-2][1]+nums[i], money[i-1][1]))
        else:
            money[i] = (max(money[i-2][0], money[i-1][0]), max(money[i-2][1]+nums[i], money[i-1][1]))
    
    return max(money[-1])


"""
Longest Palindromic Substring:
    Given a string s, return the longest palindromic substring in s. 
"""

def longestPalindrome(s: str) -> str:
    n = len(s)
    
    # initalize from left = right (odd len palindrome) 
    # or left - 1 = right (even len palindrome)
    def grow(left, right):
        while 1 <= left and right < n-1 and s[left-1]==s[right+1]:
            left -= 1
            right += 1
        return s[left,right+1]
    
    best = ''
    for i in range(n): # represents the 'center' of a possible palindrome
        pal = grow(i,i) # looking for odd length
        if len(pal) > len(best):
            best = pal
        pal = grow(i,i+1) # looking for even length
        if len(pal) > len(best):
            best = pal
    
    return best


def longestPalindromeSlow(s: str) -> str:
    n = len(s)
    known = {(i,i) for i in range(n)}
    best = s[0]
    
    if n > 1:
        flag = True
        for i in range(n-1):
            if s[i]==s[i+1]:
                known.add((i,i+1))
                if flag:
                    best = s[i:i+2]
                    flag = False
    
    for m in range(2,n):
        flag = True
        for i in range(0,n-m):
            if s[i] == s[i+m] and (i+1,i+m-1) in known:
                known.add((i,i+m))
                if flag:
                    best = s[i:i+m+1]
                    flag = False
    
    return best


"""
Palindromic Substrings:
    Return the number of substrings that are palindromes,
    where substrings are in terms of the indices, eg 'aa' has 3.
"""

def countSubstrings(s: str) -> int:
    n = len(s)
    
    # initalize from left = right (odd len palindrome) 
    # or left - 1 = right (even len palindrome)
    def grow(left, right):
        count = 0
        while 0 <= left and right < n and s[left]==s[right]:
            count += 1
            left -= 1
            right += 1
        return count
    
    total = 0
    for i in range(n): # represents the 'center' of a possible palindrome
        total += grow(i,i) # looking for odd length
        total += grow(i,i+1) # looking for even length
   
    return total

"""
Decode Ways:
    A message containing letters is encoded via standard 'A'->'1', 'B'->'2',..
    Given a string of digits, return the number of possible decodings.
"""

def numDecodings(s: str) -> int:
    ways = {str(i) for i in range(1,27)}
        
    n = len(s)
    nums = [0]*n
    for i in range(n):
        if i == 0:
            nums[0] = int(s[0] in ways)
        elif i == 1:
            nums[1] = int(s[:2] in ways) + nums[0]*(s[1] in ways)
        else:
            nums[i] = nums[i-2]*(s[i-1:i+1] in ways) + nums[i-1]*(s[i] in ways)
        
    return nums[-1]
        
"""
Coin Change:
    Given array nums of positive integers and target,
    return the minimal number of picks (with replacement)
    from array need to sum to target.
"""

def coinChange(coins: List[int], amount: int) -> int:
    best = [0] + [-1]*(amount)
    
    for i in range(1,amount+1):
        try:
            best[i] = 1 + min([best[i-coin] for coin in [small for small in coins if small <= i] if best[i-coin] > -1])
        except:
            pass

    return best[-1]
        


"""
Maximum Product Subarray:
    Given an integer array nums, find a contiguous non-empty 
    subarray within the array that has the largest product, 
    and return the product.
"""

def maxProduct(nums: List[int]) -> int:
    best = nums[0]
    n = len(nums)
    
    ans = [(nums[0],nums[0])]*n
    for i in range(1,n):
        lst = [ans[i-1][0]*nums[i], ans[i-1][1]*nums[i], nums[i]]
        ans[i] = (max(lst),min(lst))
        if ans[i][0] > best:
            best = ans[i][0]
    
    return best


"""
Word Break:
    Given a string s and a dictionary of strings wordDict, return 
    true if s can be segmented into a space-separated sequence of 
    one or more dictionary words.
"""

def wordBreak(s: str, wordDict: List[str]) -> bool:
    wordDict = {word:len(word) for word in wordDict}
    lengths = set(wordDict.values())
    
    n = len(s)
    seen = {-1}
    stack = [0]
    
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i)
            for m in lengths:
                if i+m > n:
                    break
                if s[i:i+m] in wordDict.keys():
                    if i+m == n:
                        return True
                    if i+m not in seen:
                        stack.append(i+m)
    
    return False

# test1 = [
#         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
#         ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]
#         ]
# test2 = ["leetcode", ["leet","code"]]
# test3 = ["applepenapple", ["apple","pen"]]

# wordBreak(test2[0], test2[1])


"""
Longest Increasing Subsequence:
    Given an integer array nums, return the length of the 
    longest strictly increasing subsequence.
"""
def lengthOfLIS(nums: List[int]) -> int:
        # we will iterate through nums and update/grow seq such that
        # for all increasing subsequeces subs[i] of length i (so far) 
        # seq[i-1] is the min([sub[-1] for sub in subs[i]])
        # by induction on the length of num used to build seq one sees that it is increasing
        # so we can update seq via binary search
        seq = [nums[0]]
        
        # how we update seq for the next number num
        # if num not in seq and seq[i] < num < seq[i+1]
        # then seq[i+1] = num
        def update(num):
            if num > seq[-1]:
                seq.append(num)
            else:
                left = 0
                right = len(seq)
                while left < right:
                    mid = (left+right)//2
                    if seq[mid] < num:
                        left = mid+1
                    else:
                        right = mid
                seq[left] = num
        
        for num in nums[1:]:
            update(num)
        
        return len(seq)
            

def lengthOfLISlow(nums: List[int]) -> int:
    n = len(nums)
    best = 1
    ans = [1]*n
    
    for i, num in enumerate(nums):
        if i==0:
            continue
        ans[i] = max([ans[j] for j in range(i) if nums[j] < num]+[0])+1
        if ans[i] > best:
            best = ans[i]
    return best
    
"""
Partition Equal Subset Sum:
    Given a non-empty array nums containing only positive integers, 
    find if the array can be partitioned into two subsets 
    such that the sum of elements in both subsets is equal.
"""    

def canPartition(nums: List[int]) -> bool:
    n = len(nums)
    if n==1:
        return False
    if n==2:
        return nums[0] == nums[1]
    
    s = sum(nums)
    if s % 2 == 1:
        return False
    
    # need each part to sum to target
    # so problem becomes 'is there a subset summing to target?'
    target = s//2
    
    # container for answers for numbers 0 to target
    answer = [False]*(target+1)
    answer[0] = True
    
    for num in nums:
        # need to reverse this range so that we are not taking
        # 'sums of num' multiple times in the same loop
        # eg without reversing, for num = 1, answer would all be true
        for sub in range(num,target+1)[::-1]:
            answer[sub] = answer[sub] or answer[sub-num]
        if answer[-1]:
            return True
    return False
    
       
def canPartitionSlow(nums: List[int]) -> bool:
    subprob = [{nums[0]}]*len(nums)
    for i,num in enumerate(nums):
        if i == 0:
            continue
        subprob[i] = {num + par for par in subprob[i-1]}.union({abs(par-num) for par in subprob[i-1]})
    
    return 0 in subprob[-1]   
    
