#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Math & Geometry
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq
import random
import math


        
        
"""
Happy Number:
    f: N -> N by f(x) = sum d(x,i)^2 where d(x,i) is the ith digit of x
    For given n, determine if n flows to 1 or not.
    There are cycles 20 -> 4 -> 16 -> 37 -> 58 -> 89 -> 145 -> 42 -> 20
    f(x) < 100*log(x), so orbits are bounded, so must be 
    eventually periodic (or stationary) 
"""

def isHappy(n: int) -> bool:
    def f(n):
        return sum([x**2 for x in map(int,list(str(n)))])

    out = defaultdict(int)
    
    while True:
        n = f(n)
        if n == 1:
            return True
        if out[n] == 1:
            return False
        out[n]+=1

"""
Plus One:
    Given a list of int in [0,9], thought of as digits of a number x.
    Return a list of int representing x+1.
"""

def plusOne(digits: List[int]) -> List[int]:        
    num = int(''.join([str(x) for x in digits]))+1
    return [int(c) for c in str(num)]