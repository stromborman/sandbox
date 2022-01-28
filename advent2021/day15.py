# -*- coding: utf-8 -*-
"""
Spyder Editor

Solution to Advent of Code 2021: Day 15
"""
from heapq import heappop, heappush
import igraph as ig
import numpy as np


input = np.genfromtxt('input15', delimiter=1 ,dtype=int)
test = np.genfromtxt('input15_test', delimiter=1,dtype=int)


def make_graph(mat):
    size = len(mat)
    gr = ig.Graph.Lattice([size,size], circular=False)
    gr.vs['risk'] = mat.flatten()
    return gr


def distance(graph, start, end):
    gr = graph
    gr.vs['dist'] = np.inf
    gr.vs(start)['dist'] = 0
    gr.vs['done'] = False
    gr.vs(start)['done'] = True
    gr.vs['prev'] = -1

    pq = []
    heappush(pq, (0, start))

    while len(pq) > 0 and gr.vs(end)['done'][0] is False:
        # print(len(pq) > 0 and gr.vs(end)['done'][0]==False)
        _, vert = heappop(pq)
        # print('poping '+str(vert))
        for n_vert in gr.neighbors(vert):
            if gr.vs(n_vert)['done'][0] is False and \
                gr.vs(vert)['dist'][0] + gr.vs(n_vert)['risk'][0] \
                    < gr.vs(n_vert)['dist'][0]:
                # print('considering '+str(n_vert))
                gr.vs(n_vert)['dist'] = gr.vs(vert)['dist'][0] + gr.vs(n_vert)['risk'][0]
                gr.vs(n_vert)['prev'] = vert
                heappush(pq, (gr.vs(n_vert)['dist'][0], n_vert))
                # print(pq)
        gr.vs(vert)['done'] = True

    return gr.vs(end)['dist']

print('Answer to part1: '+str(distance(make_graph(input),0,9999)[0]))


def block(mat):
    blocks = np.empty((5,5), dtype= np.ndarray)
    for i in range(5):
        for j in range(5):
            blocks[i,j] = (mat - 1 + i + j)%9 + 1

    rowblocks = []
    for i in range(5):
        rowblocks = rowblocks + [np.hstack([blocks[i,j] for j in range(5)])]

    return np.vstack([rowblocks[i] for i in range(5)])

print('Answer to part2: '+str(distance(make_graph(block(input)),0,249999)[0]))