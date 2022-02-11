#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 8
"""

import pandas as pd
import numpy as np

input = pd.read_csv('input08', header=None, sep='\s', engine='python')
input = input.drop(columns = [10]).set_axis(np.arange(14),axis=1)

test = pd.read_csv('input08_test', header=None, sep='\s', engine='python')
test = test.drop(columns = [10]).set_axis(np.arange(14),axis=1)

def to_len(df):
    df1 = pd.DataFrame()
    for j in range(df.shape[1]):
        df1[j] = df[j].str.len()
    return df1
    
def count_ez(df):
    df1 = to_len(df).iloc[:,10:14]
    return df1.isin([2,3,4,7]).sum().sum()

print('Answer to day8/part1: '+ str(count_ez(input)))

def decode(ser):
    s= ser.iloc[0:10].apply(sorted).str.join(sep='')
    sl = s.apply(len)
    c= pd.Series(dtype=str, index=range(10))

    c[1]= s[sl==2].iloc[0]
    c[7]= s[sl==3].iloc[0]
    c[4]= s[sl==4].iloc[0]
    c[8]= s[sl==7].iloc[0]


    cf = c[1]
    bd = c[4].translate({ord(i):None for i in cf})

    l5 = s[sl==5]

    cf_bool = l5.apply(lambda x: all(z in x for z in cf))
    bd_bool = l5.apply(lambda x: all(z in x for z in bd))

    c[5] = l5[bd_bool].iloc[0]
    c[3] = l5[cf_bool].iloc[0]
    c[2] = l5[(~bd_bool)&(~cf_bool)].iloc[0]

    l6 = s[sl==6]

    cf_bool6 = l6.apply(lambda x: all(z in x for z in cf))
    bd_bool6 = l6.apply(lambda x: all(z in x for z in bd))

    c[0] = l6[~bd_bool6].iloc[0]
    c[6] = l6[~cf_bool6].iloc[0]
    c[9] = l6[(bd_bool6)&(cf_bool6)].iloc[0]

    dict = {c[i]: i for i in range(10)}
    
    out = ser.iloc[10:14].apply(sorted).str.join(sep='').map(dict)\
        .to_numpy()@[10**(3-i) for i in range(4)]
    return out

def sum_of_outs(df):
    out = 0
    for i in range(len(df)):
        out = out + decode(df.iloc[i])
    return out

print('Answer to day8/part2: '+ str(sum_of_outs(input)))
