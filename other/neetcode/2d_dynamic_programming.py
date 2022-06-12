#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
2-D Dynamic Programming
"""

from collections import defaultdict, deque
from typing import List, Optional
from copy import deepcopy
import numpy
import heapq
import math
import random
        
        
"""
Unique Paths:
    Number of distinct path from (0,0) to (m-1,n-1) using (1,0) and (0,1) steps

    Answer is just (n-1)+(m-1) choose m-1.  Below is practice using dynamical
"""


def uniquePaths(m: int, n: int) -> int:
    ans = numpy.full((m,n),1)
    for row in range(m-1)[::-1]:
        for col in range(n-1)[::-1]:
            ans[row][col] = ans[row+1][col] + ans[row][col+1]
    return ans[0][0]



# for i in range(10):
#     n = random.randint(1,10)
#     m = random.randint(1, 10)
#     print(n,m, uniquePaths(m, n) == math.comb(n+m-2,m-1)) 
    
"""
Longest Common Subsequence:
    Given two strings text1 and text2, return the length of their 
    longest common subsequence.
    
    TIL: y = [x]*n consists of n shallow copies of x (if x is mutable)
    but y = [x for i in range(n)] makes deepcopies
"""

def longestCommonSubsequence(text1: str, text2: str) -> int:
    n, m = len(text1), len(text2)

    ans = [[0]*(m+1) for _ in range(n+1)]

    for i,c1 in enumerate(text1):
        for j,c2 in enumerate(text2):
            ans[i+1][j+1] = max(ans[i][j+1],ans[i+1][j])
            if c1==c2 and ans[i][j]+1 >= ans[i+1][j+1]:
                ans[i+1][j+1] = ans[i][j]+1
    return ans[-1][-1]


"""
Best Time to Buy and Sell Stock with Cooldown:
    You are given an array prices where prices[i] is the price of a 
    given stock on the ith day.  Find the maximum profit you can achieve. 
    You can only hold a single stock and cannot buy/sell the same day.
    After selling need to cooldown 1 day.
"""

def maxProfitD(prices: List[int]) -> int:
    best = {'buy':-prices[0],'hold':-prices[0],'sell':0,'wait':0}
    for price in prices:
        new = {}
        new['buy'] = best['wait'] - price
        new['hold'] = max(best['buy'],best['hold'])
        new['sell'] = max(best['buy'],best['hold']) + price
        new['wait'] = max(best['sell'],best['wait'])
        best = new
    return max(best.values())


# This is just a list version (runs faster for LC)
# Dict version much easier to read for a human

def maxProfit(prices: List[int]) -> int:
    best = [-prices[0],-prices[0],0,0]
    for price in prices:
        best = [best[3] - price,max(best[0],best[1]),
                max(best[0],best[1]) + price,max(best[2],best[3])]
    return max(best)
   
    
# print(maxProfit([1,2,3,0,2]))
    
"""
Coin Change 2:
    Return the number of paritions of amount
    using the positive ints from coins (with replacement)
"""

def change(amount: int, coins: List[int]) -> int:
    coins = sorted(coins)
    ans = [[1]+[0]*(amount) for coin in coins]
    
    # ans[i][tar] is the number of ways to get a sum of tar
    # using the coins in coins[:i+1]
    
    for tar in range(1,amount+1):
        for i, coin in enumerate(coins):
            if i > 0:
                ans[i][tar] = ans[i-1][tar]
            if coin <= tar:
                ans[i][tar] += ans[i][tar-coin]
    
    return ans[-1][-1]

"""
Interleaving String:
    Given strings s1, s2, and s3, find whether s3 is formed 
    by an interleaving (think ripple shuffle) of s1 and s2.
"""

def isInterleave(s1: str, s2: str, s3: str) -> bool:
    l1, l2, l3 = len(s1), len(s2), len(s3)
    
    if l1 + l2 != l3:
        return False
    
    if l3 == 0:
        return True
    
    if ''.join(sorted(s1+s2)) != ''.join(sorted(s3)):
        return False
    
    work = {(0, 0)}
    new = set([])
    for c in s3:
        new = set([])
        for pair in work:
            if pair[0] < l1:
                if s1[pair[0]] == c:
                    new.add((pair[0]+1,pair[1]))
            if pair[1] < l2:
                if s2[pair[1]] == c:
                    new.add((pair[0],pair[1]+1))
        if new == {[]}:
                return False
            
        work = new
    return True
        









