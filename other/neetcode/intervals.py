#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Intervals
"""

from collections import defaultdict, deque
from typing import List, Optional, Tuple
from copy import deepcopy
import heapq
import random


        
        
"""
Meeting Rooms:
    Given a list of half open intervals [a,b), return True if their intersection is empty
"""

def canAttendMeetings(intervals: List[Tuple]) -> bool:
    starts = []
    ends = []
    for i,item in enumerate(intervals):
        for j in range(i):
            if starts[j] <= item[0] < ends[j]:
                return False
            if starts[j] < item[1] <= ends[j]:
                return False
        starts.append(item[0])
        ends.append(item[1])
    return True

# for intervals in [[(0,30),(5,10),(15,20)],[(5,8),(9,15)]]:
#     print(canAttendMeetings(intervals))  

"""
Meeting Rooms II:
    Given a list of half open intervals [a,b), return max of the function
                                         f(x) = |{intervals : x in intervals}|
"""

def minMeetingRooms(intervals: List[Tuple]) -> int:
    starts = [[]]
    ends = [[]]
    rooms = [0]
    for item in intervals:
        print(item)
        r = 0
        while r < len(rooms):
            flag = True
            for j in range(len(starts[r])):
                if starts[r][j] <= item[0] < ends[r][j]:
                    flag = False
                    print('conflict when checking room',r)
                    break
                elif starts[r][j] < item[1] <= ends[r][j]:
                    flag = False
                    print('conflict when checking room',r)
                    break
                elif item[0] <= starts[r][j] < item[1]:
                    flag = False
                    print('conflict when checking room',r)
                    break
            if flag:
                print('assigning to room',r)
                starts[r].append(item[0])
                ends[r].append(item[1])
                break
            else:
                r += 1
        if r == len(rooms):
            print('opening new room', r)
            rooms.append(r)
            starts.append([item[0]])
            ends.append([item[1]])
        
    return len(rooms)
    
# test = []
# for i in range(20):
#     s = random.randint(8, 17)
#     e = s+random.randint(1, 5)
#     test.append((s,e))
# print(minMeetingRooms(test))
# print(sorted(test))
# print(max([(x,sum([item[0] <= x < item[1] for item in test])) for x in range(24)],key=lambda y:y[1]))
