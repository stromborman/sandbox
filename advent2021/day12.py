#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 12
"""

import igraph as ig
import re
import numpy as np

def edges(x):
    with open(x) as f:
        lines = f.read().splitlines()
    r = re.compile(r'(\w+)\-(\w+)')
    v0 = [ r.sub(r'\1', line) for line in lines ]
    v1 = [ r.sub(r'\2', line) for line in lines ]
    return [ (v0[i], v1[i]) for i in range(len(v0)) ]

inp = edges('input12')
t0 = edges('input12_test_0')
t1 = edges('input12_test_1')
t2 = edges('input12_test_2')

def g(x):
    g = ig.Graph.TupleList(x)
    g.vs['small'] = False
    r = re.compile(r'^[a-z]{2}$')
    for x in list(filter(r.match, g.vs()['name'])):
        v = g.vs.find(name=x)
        g.vs(v.index)['small'] = True
    return g