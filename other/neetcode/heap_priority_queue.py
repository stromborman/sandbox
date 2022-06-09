#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Heap / Priority Queue
"""

from collections import defaultdict, deque
from typing import List, Optional
from copy import deepcopy
import heapq


        
        
"""
Kth Largest Element in a Stream
"""

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        heapq.heapify(nums)
        while (len(nums) > self.k):
            heapq.heappop(nums)
        self.maxheap = nums
    
    def add(self, val: int) -> int:
        if len(self.maxheap) >= self.k:
            heapq.heappushpop(self.maxheap,val)
        else:
            heapq.heappush(self.maxheap,val)
        return self.maxheap[0]
    
# tests = [
#          [["KthLargest","add","add","add","add","add"], [[3,[4,5,8,2]],[3],[5],[10],[9],[4]]],
#          [["KthLargest","add","add","add","add","add"], [[1,[]],[-3],[-2],[-4],[0],[4]]]
#         ]

# for test in tests:
#     kth = KthLargest(test[1][0][0], test[1][0][1])
#     toprint = []
#     for i in range(1,6):
#         toprint.append(kth.add(test[1][i][0]))
#     print(toprint)

"""
Last Stone Weight
"""

def lastStoneWeight(stones: List[int]) -> int:
    heap = [-stone for stone in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        y = -heapq.heappop(heap)
        x = -heapq.heappop(heap)
        if y > x:
            heapq.heappush(heap,x-y)
    if heap:
        return -heap[0]
    else:
        return 0
    
    
    
"""
K Closest Points to Origin:
    Given an array of points in R^2 return the K closest points to the origin
"""

def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    # One liner
    # return heapq.nsmallest(k,points,key=lambda x:x[0]**2+x[1]**2)
    
    heap = []
    for point in points:
        norm = point[0]**2 + point[1]**2
        if len(heap) < k:
            heapq.heappush(heap,(-norm,point))
        elif len(heap) == k:
            heapq.heappushpop(heap,(-norm,point))
    return [point for n,point in heap]


