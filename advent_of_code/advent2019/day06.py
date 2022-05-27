#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2019: Day 06
https://adventofcode.com/2019/day/6

The problem takes place in the context of a tree with root: COM

The input data is a list of entries with the form: ABC)DEF
where ABC and DEF are vertices and there is an edge ABC -> DEF
"""

from collections import deque


class DirectedGraph:
    def __init__(self,verts:set,oEdges:dict,root=None):
        self.verts = verts # set of vertices
        self.oEdges = oEdges # a dict with keys in verts and values set of verts
        self.root = root
        self.level = self.compLevel(self.oEdges,self.root)
        self.iEdges = self.inEdges(self.verts, self.oEdges)
       
    @staticmethod
    def compLevel(oEdges,root):
        level = {root:0}
        queue = deque([root])
        
        while queue:
            work = queue.popleft()
            if work in oEdges.keys():
                for vert in oEdges[work]:
                    level[vert] = level[work] + 1
                    queue.append(vert)
        return level
        
    @staticmethod
    def inEdges(verts, oEdges):
        dic = {v:set() for v in verts}
        for v in verts:
            for w in oEdges[v]:
                dic[w].add(v)
        return dic


def make_graph(filename):
    with open(filename) as file:
        edges = [item.split(')') for item in file.read().splitlines()]
        edge_dic = {}
        verts = set()
        for edge in edges:
            verts = verts.union(set(edge))
            if edge[1] not in edge_dic.keys():
                edge_dic[edge[1]] = set()   
            if edge[0] in edge_dic.keys():
                edge_dic[edge[0]].add(edge[1])
            else:
                edge_dic[edge[0]] = set([edge[1]])
        return DirectedGraph(verts, edge_dic,'COM')
    
test = make_graph('input06test')    
real = make_graph('input06')
    
"""
For part 1: We need to sum over all vertices in the graph the distance to the root vertex.
"""


def sum_of_depths(dir_graph):
    running_sum = 0
    for key,value in dir_graph.level.items():
        running_sum += value
    return running_sum

print('Answer to part1:', sum_of_depths(real))


"""
For part 2: Thinking of the graph as undirected, we need to compute the distance 
from (parent of) 'YOU' to (parent of) 'SAN'
"""

def dist(dir_graph,vert1,vert2):
    def path_up(vert):
        geneology = [vert]
        work = vert
        while dir_graph.iEdges[work]:
            work = list(dir_graph.iEdges[work])[0]
            geneology.append(work)
        return [(dir_graph.level[item],item) for item in geneology]
    
    common_ancestors = list(set(path_up(vert1)).intersection(path_up(vert2)))
    common_ancestors.sort(key=lambda x:x[0])
    meet_level = common_ancestors[-1][0] + 1

    return dir_graph.level[vert1] + dir_graph.level[vert2] - 2*meet_level
    
    
print('Answer to part2:', dist(real,'YOU','SAN'))