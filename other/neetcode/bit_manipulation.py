#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Bit Manipulation
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq
import random
import math


        
        
"""
Number of 1 Bits
"""

def hammingWeight(n: int) -> int:
    return bin(n).count('1')
    

"""
Counting Bits:
    Given an integer n, return an array ans of length n + 1 such that for each i 
    (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
"""

def countBits(n: int) -> List[int]:
        if n == 0:
            return [0]
        
        full_loops = int(math.log2(n))+1
        ans = [0,1]+[0]*(2**full_loops)
        
        for i in range(1,full_loops):
            for j in range(0,2**(i-1)):
                ans[2**i + 2*j] = ans[2**(i-1) + j]
                ans[2**i + 2*j + 1] = ans[2**(i-1) + j] + 1

        return ans[:n+1]
                
def countBitsSameButCleaner(n: int) -> List[int]:
    ans = [0]*(n+1)
    for i in range(1,n+1):
        # n>>k converts n it bin, cuts off the k right most bits and returns result as int
        # equivalent to int(n/2^k)
        ans[i] = ans[i>>1] + (i%2)  
    return ans
        
"""
Reverse Bits
"""

def reverseBits(n: int) -> int:
    s = f'{n:032b}'
    return int('0b'+s[::-1],base=2)

"""
Missing Number:
    Given an arr of len n, containing unique elements of [0,n], determine
    the missing element
"""

def missingNumber(nums: List[int]) -> int:
    n = len(nums)
    if n==1:
        return nums[0]^1
    top = int(math.log2(n))+1
    nums = nums + list(range(n+1,2**top))
    out = nums[0]
    for num in nums[1:]:
        out = out^num
    return out

"""
Single Number:
    List of nums with one unique num and rest appear twice.  Recover the unique num.
"""

def singleNumber(nums: List[int]) -> int:
    out = nums[0]
    for num in nums[1:]:
        out = out ^ num
    return out

"""
Sum of Two Integers:
    Given two integers a and b, return the sum of the two integers 
    without using the operators + and -.
"""

def add(a,b): # a,b >=0 or a,b <= 0
    i = 0
    while b!=0 and i < 10:
        i+= 1
        c = (a&b)<<1
        a = a^b
        b = c
    return a

def sub(a,b): # a >= b >= 0
    i = 0
    while b!=0 and i < 10:
        i+= 1
        c = (~a&b)<<1
        a = a^b
        b = c
    return a
    
def getSum(a: int, b: int) -> int:  
    if a*b >= 0:
        return add(a,b)
    
    # ensure |a| >= |b|
    if abs(a) < abs(b):
        return getSum(b, a)
    
    if a < 0:
        return -sub(abs(a),b)
    
    else:
        return sub(a,abs(b))

    

