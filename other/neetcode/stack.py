#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Stack
"""

from typing import List
from collections import defaultdict
import math
import timeit


"""
Valid Parentheses:
    Given a string s containing just the characters 
    '(', ')', '{', '}', '[' and ']', determine if the input string is valid
"""

def isValid(s: str) -> bool:
     open_close = {'(':')','{':'}','[':']'}
     stack = []
     
     for c in s:
         if c in open_close.keys():
             stack.append(c)
         else:
             try:
                 par = stack.pop()
             except:
                 return False
             
             if open_close[par] != c:
                 return False
     
     return len(stack) == 0 

"""
Evaluate Reverse Polish Notation:
    Evaluate the value of an arithmetic expression in Reverse Polish Notation.
    (Given as an array of strings, each an int or operation)
    RPN means operators come after inputs, eg 3,4,+,5,- is (3+4)-5 
"""  

def evalRPN(tokens: List[str]) -> int:
    stack = []
    opers = frozenset(['+','-','*','/'])
    mem = None
    for op in tokens:
        if op not in opers: 
            stack.append(op)
        else:
            b = stack.pop()
            a = stack.pop()
            if op != '/':
                mem = str(eval(a+op+b))
            else:
                mem = str(math.trunc(eval(a+op+b)))
            stack.append(mem)
    return int(stack.pop())

"""
Generate Parentheses:
    A function to generate all combinations of well-formed parentheses using n pairs
"""   


def generateParenthesisRecursive(n: int) -> List[str]:
    if n == 1:
        return ['()']
    ans = []
    stack = [('(',1,1)]
    while stack:
        pars,num_open,par_sum = stack.pop()
        if num_open + 1 == n:
            ans.append(pars+'('+')'*(par_sum+1))
        else:
            stack.append((pars+'(',num_open+1,par_sum+1))
        
        if par_sum > 1:
            stack.append((pars+')',num_open,par_sum-1))
        elif par_sum == 1:
            ans = ans + [pars+')'+done_pars for done_pars in generateParenthesisRecursive(n-num_open)]
            
    return ans

# Backtracking is 10x faster
def generateParenthesis(n: int) -> List[str]:

        stack = []
        res = []
        
        def backtrack(open_count, close_count):
            # print('BACK', open_count,close_count)
            
            if (open_count == n == close_count):
                res.append("".join(stack))
                # print('res now:', res)
                return
            
            if open_count < n:
                stack.append("(")
                # print('stack:', ''.join(stack))
                backtrack(open_count + 1, close_count)
                stack.pop()
                # print('rewindstack:', ''.join(stack))
            
            if close_count < open_count:
                stack.append(")")
                # print('stack:', ''.join(stack))
                backtrack(open_count, close_count + 1)
                stack.pop()
                # print('rewindstack:', ''.join(stack))
        
        backtrack(0, 0)
        
        return res

"""
Daily Temperatures:
    Given an array of integers temperatures represents the daily temperatures, 
    return an array answer such that answer[i] is the number of days you 
    have to wait after the ith day to get a warmer temperature. 
    If there is no future day for which this is possible, 
    keep answer[i] == 0 instead.
"""

def dailyTemperatures(temperatures: List[int]) -> List[int]:
    answer = [0]*len(temperatures)
    stack = [(0,temperatures[0])]
    for i, temp in enumerate(temperatures[1:]):
        i = i+1
        while len(stack) > 0 and temp > stack[-1][1]:
            j,_ = stack.pop()
            answer[j] = i-j
        stack.append((i,temp))
    return answer



"""
Car Fleet:
    There are n cars going to the same destination along a one-lane road. 
    The destination is target miles away. 
    You are given two integer array position and speed, both of length n, 
    where position[i] is the position of the ith car and speed[i] is the 
    speed of the ith car (in miles per hour).

    A car can never pass another car ahead of it, but it can catch up to it and 
    drive bumper to bumper at the same speed. The faster car will slow down 
    to match the slower car's speed. The distance between these 
    two cars is ignored (i.e., they are assumed to have the same position).

    A car fleet is some non-empty set of cars driving at the same position 
    and same speed. Note that a single car is also a car fleet.

    If a car catches up to a car fleet right at the destination point, 
    it will still be considered as one car fleet.
    
    Return the number of car fleets that will arrive at the destination.
    
    Key observation: Even with the potential slowdowns due to 
    intermediate merges, if c1 < c2 < c3 in positions, c2 catches
    c3 (based on inital speeds), and c1 catches c3 based on inital speeds,
    then c1 still catches c3 even if c2 slows c3 down.
"""


def carFleet(target: int, position: List[int], speed: List[int]) -> int:
    pv = sorted(list(zip(position,speed)), key= lambda x:-x[0])
    
    stack = []
    
    for p,v in pv: #going through the pv list starting at cars closest to target
        target_time = (target-p)/v # time the car reaches the target at initial speed
        
        # Not adding to the stack means the car merges before the end
        if not stack or target_time > stack[-1]: 
            stack.append(target_time)
                
    return len(stack)
