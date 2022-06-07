#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
1-D Dynamic Programming
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