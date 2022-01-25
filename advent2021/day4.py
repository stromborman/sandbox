#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:24:38 2022

@author: cadoi
"""

import pandas as pd
import numpy as np

nums = pd.read_csv('~/advent2021/input4', dtype = int, header=None, nrows=1)
nums = nums.T.to_numpy()
boards = pd.read_csv('~/advent2021/input4', header=None, skiprows=1, sep='\s+')

B = np.full(100, fill_value= None)
# B = np.empty(100)#, dtype=np.array)
for i in range(0,100):
    B[i] = boards.iloc[5*i:5*i+5].reset_index(drop=True).to_numpy()
 

def col_to_check(B):
    return np.concatenate((B.sum(axis=0), B.sum(axis=1)))

def bingo(card, numlist=nums):
    winturn = 0
    finalnum = 0
    markedcard = np.full((5,5), fill_value = None)
    score = 0
    for i, n in enumerate(numlist):
        card = np.where(card == n, -1, card)
        if np.isin(-5, col_to_check(card)):
            winturn = i
            markedcard = np.where(card == -1, 0, card)
            finalnum = int(n)
            score = int(markedcard.sum() * finalnum)
            break
    return [winturn, markedcard, finalnum, score]    

win = pd.DataFrame(
    [ [i, bingo(B[i], nums)[0], bingo(B[i], nums)[3] ] for i in range(100) ] )

win = win.sort_values(by=[1], ignore_index=True)

# print('Fastest card is number ' + str(win[0][0]) + ' that wins on turn '+
#       str(win[1][0]) + ' with score '
#        + str(win[2][0]))

print('Answer to day4/part1: '+str(win[2][0]))

# print('Slowest card is number ' + str(win[0][99]) + ' that wins on turn '+
#       str(win[1][99]) + ' with score '
#        + str(win[2][99]))   

print('Answer to day4/part2: '+str(win[2][99]))