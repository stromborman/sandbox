#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 12
"""

import re
import igraph as ig
import matplotlib.pyplot as plt


# processes raw edge file and returns a list of 2-tuples representing the edges
def edges(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
    reg = re.compile(r'(\w+)\-(\w+)')
    ver0 = [ reg.sub(r'\1', line) for line in lines ]
    ver1 = [ reg.sub(r'\2', line) for line in lines ]
    return [ (ver0[i], ver1[i]) for i in range(len(ver0)) ]

# variable names for processed edge files
inp = edges('input12')
t1 = edges('input12_test_1')
t2 = edges('input12_test_2')

# takes an edge list from edges and creates an igraph object
def g(edge_list):
    gr = ig.Graph.TupleList(edge_list)
    gr.vs['big'] = False
    gr.vs['s/e'] = False
    r = re.compile(r'[A-Z]')
    # marks the big caves
    for na in list(filter(r.match, gr.vs()['name'])):
        v = gr.vs.find(name=na)
        gr.vs(v.index)['big'] = True
    s = gr.vs.find(name='start').index
    gr.vs(s)['s/e'] = True
    e = gr.vs.find(name='end').index
    gr.vs(e)['s/e'] = True
    return gr

# plot the graph
def plotg(g):
    layout = g.layout("kk")
    fig, ax = plt.subplots()
    visual_style = {}
    visual_style["vertex_label"] = list(enumerate(g.vs["name"]))
    visual_style["layout"] = layout
    visual_style["vertex_label_size"] = 15
    visual_style["target"]=ax
    ig.plot(g, **visual_style)

def no_repeats(graph, vertex, path):
    return graph.vs(vertex)['big']==[True] or vertex not in path

# iteratively search for paths from a to b one step at a time
def find_paths(g, a, b, path=[], debug=False):
    # add a to current path
    path = path + [a]
    # if current path now at b record the path
    if a == b:
        return [path]
    #list of paths we are building
    paths = []
    for v in g.neighbors(a):
        if debug is True:
            print('current path '+ str(path) + ' move to '+str(v))
            print(g.vs(v)['big']==[True] or v not in path)
        # check if that v next to is not a small cave already on path
        if no_repeats(g, v, path):
            if debug is True:
                print(path + [v])
            # now iterate above, so we add v to path and check its neighbors
            v_paths = find_paths(g, v, b, path)
            for v_path in v_paths:
                # record completed paths to our list paths
                paths.append(v_path)
    return paths

def count_paths(x):
    s = g(x).vs.find(name='start').index
    e = g(x).vs.find(name='end').index
    return len(find_paths(g(x),s,e))

print('Answer to part1: '+str(count_paths(inp)))

def allow_repeat(g,v,path):
    return g.vs(v)['big']==[False] and g.vs(v)['s/e']==[False] and v in path

def find_crazy_paths(g, a, b, path=[]):
    # add a to current path
    path = path + [a]
    # if current path now at b we stop and record the path
    if a == b:
        return [path]
    #list of paths we are building
    paths = []
    for v in g.neighbors(a):
        if allow_repeat(g,v,path):
            # if v is repeat small cave, allow it
            # using find_paths here so we avoid more repeats
            v_paths = find_paths(g, v, b, path)
            for v_path in v_paths:
                paths.append(v_path)
        elif no_repeats(g, v, path):
            v_paths = find_crazy_paths(g, v, b, path)
            for v_path in v_paths:
                paths.append(v_path)
    return paths

def count_crazy_paths(x):
    s = g(x).vs.find(name='start').index
    e = g(x).vs.find(name='end').index
    return len(find_crazy_paths(g(x),s,e))

print('Answer to part2: '+str(count_crazy_paths(inp)))
               