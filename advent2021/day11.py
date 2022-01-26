#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 11
"""

import igraph as ig
import numpy as np
from queue import Queue

input = np.genfromtxt('input11', delimiter=1 ,dtype=int)
test = np.genfromtxt('input11_test', delimiter=1,dtype=int)
test0 = np.genfromtxt('input11_test_small', delimiter=1,dtype=int)



def make_graph(x):
    g = ig.Graph.Lattice([10,10], circular=False)

    newedge = [(i+10*j,i+11+10*j) for i in range(9) for j in range(9)] + \
    [(i+10*j, i+9+10*j) for i in range(1,10) for j in range(9)]
    
    g.add_edges(newedge)
    g.vs['energy'] = x.flatten()
    g.vs['flashed'] = False
    return g

def plus1(gr):
    gr.vs['energy'] = [x+1 for x in gr.vs['energy']]

def flash(n, gr):
    gr.vs[n]['flashed'] = True
    nbh = gr.neighbors(n)
    for x in nbh:
        gr.vs[x]['energy'] = gr.vs[x]['energy'] + 1 
        
def reset(gr):
    j=0
    for v in gr.vs.select(flashed=True):
        j = j+1
        gr.vs[v.index]['energy'] = 0
    gr.vs['flashed'] = False
    return j


def prop(gr):
    initial = gr.vs.select(energy = 10)
    keeptrack = {v.index for v in initial}
    if len(keeptrack) > 0:
        q = Queue()
        for v in initial: 
            q.put(v.index)
        while q.empty() == False:
            v = q.get()
            if gr.vs[v]['flashed'] == False:
                flash(v,gr)
                for w in gr.neighbors(v):
                    if gr.vs[w]['flashed'] == False and gr.vs[w]['energy'] >= 10:
                        q.put(w)

    
def cycle(gra, n, c=0, pt=False):
    gr = gra
    d = c
    for i in range(1,n+1):
        plus1(gr)
        prop(gr)
        f = reset(gr)
        d = d+f
        if pt==True: 
            print('in round '+str(i)+' there were '+str(f)+' flashes')
            print(np.array(gr.vs['energy']).reshape((10,10)))
    print('during '+str(n)+' rounds there were '+str(d)+' total flashes')
    return d    
    

print('Answer to part1: '+str(cycle(make_graph(input),100)))


def when(gra):
    did_it_happen = False
    n=0
    gr = gra
    while did_it_happen == False:
        n = n+1
        plus1(gr)
        prop(gr)
        f = reset(gr)
        if f == 100: did_it_happen = True
    return n

print('Answer to part2: '+str(when(make_graph(input))))