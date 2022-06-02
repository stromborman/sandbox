#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Sliding Window
"""

from typing import List
from collections import defaultdict


"""
Best Time to Buy and Sell Stock:
    For array of prices, compute max(arr[i]-arr[j] for i >=j)
"""

def maxProfit(prices: List[int]) -> int:
    best = 0
    buy = prices[0]
    
    for price in prices[1:]:
        if price < buy:
            buy = price
        if price - buy > best:
            best = price - buy
    return best

"""
Longest Substring Without Repeating Characters:
"""

def lengthOfLongestSubstring(s: str) -> int:
    best = 0
    start = 0
    char_loc = {}
    
    for i,char in enumerate(s):
        if char not in char_loc or char_loc[char]<start:
            char_loc[char]=i
            if i-start + 1> best:
                best = i-start + 1
            
        else:
            old = char_loc[char]
            start = old+1
            char_loc[char] = i
            
    return best
                

"""
Longest Repeating Character Replacement:
    Return the length of the longest substring containing the same letter you can get 
    after changing at most k subsitutions.
"""

def characterReplacementBrute(s: str, k: int) -> int:
    def helper(s,k,let):
        best = 0
        start = 0
        swaps = []
        count = 0

        for i,char in enumerate(s):
            if char != let:
                swaps.append(i)
                if count < k:
                    count += 1
                    best = max(best, i - start +1)
                else:
                    start = swaps[0]+1
                    swaps = swaps[1:]
            else:
                best = max(best, i - start +1)
        
        return best

    best = 0
    
    for let in ''.join(set(s)):
        best = max(helper(s,k,let), best)

    return best
                
        
def characterReplacement(s: str, k: int) -> int:      
    char_counts = defaultdict(int)
    start = 0
    best = 0
    
    for end, char in enumerate(s):
        char_counts[char]+=1
        
        # this is length of window - frequency of most common, aka num of subs to make
        if end - start + 1 - max(char_counts.values()) > k:
            char_counts[s[start]] -= 1
            start += 1
        
        best = max(end-start + 1, best)
    
    return best

    
    
    
    
    
    
    
    
    
