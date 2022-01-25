#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 09:38:57 2022

@author: cadoi
"""

import pandas as pd

df  = pd.read_csv('~/advent2021/input2', names=['move'])

df['encode'] = df['move']

df['encode'].replace(regex={r'(up)\s(\d+)':'down -'r'\2'}, inplace=True)

df['down'] = 0

df['forward'] = 0

df['down'] = df['encode'].str.extract(r'down\s(-\d+|\d+)')
    
df['forward'] = df['encode'].str.extract(r'forward\s(\d+)')

df = df.fillna(0)

final = df[['down', 'forward']].astype(int)


final_depth = final['down'].sum()
final_for = final['forward'].sum()

print('Answer to day2/part1: (depth, forward, product) = '+ \
      str((final_depth, final_for, final_depth*final_for)))
    
dfaim = final.rename(columns={'down':'dAim'})
    
dfaim['aim'] = dfaim.dAim.cumsum()

depth_v2 = (dfaim.aim * dfaim.forward).sum()

print('Answer to day2/part2: (depth, forward, product) = '+ \
      str((depth_v2, final_for, depth_v2*final_for)))







