#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

import heapq
import binarytree as bt
from collections import deque
import networkx as nx  
            
"""
#9.1 Intersection of lists 
"""             

def fun91(tup): #lst is a 2-tuple of lists
    sets = [set(x) for x in tup]
    return [i for i in sets[0] if i in sets[1]]

def test91(n=20):
    for i in range(10):
        lst1 = [random.randint(-10*n,10*n) for i in range(n)]
        set1 = set(lst1)
        lst2 = [random.randint(-10*n,10*n) for i in range(n)]
        set2 = set(lst2)
        if set(fun91([lst1,lst2])) != set1.intersection(set2):
            print('Failure for', lst1, lst2)
            print('Got', fun91([lst1,lst2]),'should have got', set1.intersection(set2))
            break
    print('Passed tests for #9.1')
    

"""
#9.2 list of ints to max triple product 
""" 

def fun92(lst:list)->int:
    big = heapq.nlargest(3, lst)
    neg = heapq.nsmallest(2, lst)  
    return big[0]*max(big[1]*big[2], neg[0]*neg[1])

def test92(n=20):
    for i in range(10):
        lst = [random.randint(-20*n,20*n) for i in range(n*10)]
        whatigot = fun92(lst)
        slst = sorted(lst)
        whatiwant = slst[-1]*max(slst[0]*slst[1], slst[-3]*slst[-2])
        if whatigot != whatiwant:
            print('Failure for', lst)
            print('Got', whatigot,'should have got', whatiwant)
            break
    print('Passed tests for #9.2')
    

 
"""
#9.3: list of elements in R^2, return k closest to origin
""" 

def fun93(lst:list,k:int) -> list:
    return sorted(test, key=(lambda x: x[0]**2 + x[1]**2))[:k]
    
    
"""
#9.4: nxn matrix with columns and rows sorted, return k-th largest element
""" 
import math

def fun94(arr,k):
    l = math.ceil(-.5 + math.sqrt(-1+ 2*k)) - 1
    lst = [arr[i,l-i] for i in range(0,l+1)]
    w = l*(l+1)/2
    return sorted(lst)[k-w-1]

"""
#9.5: for arr, return max(0, sum(arr[i:j])) over all i,j 
""" 

def fun95(arr):
    l = len(arr)
    best = 0
    cur = 0
    for i in range(l):
        cur += arr[i]
        if cur > best:
            best = cur
        if cur < 0:
            cur = 0
    return best

# test = [random.randint(-10, 100) for i in range(100)]
# fun95(test)

"""
#9.6: function to determine if binary tree is mirror image of itself
(least typing: run DFS HereLeftRight and HereRightLeft and check for reversed lists)
""" 
import copy

def mirror(tree):
    newtree = copy.deepcopy(tree)
    if tree == None:
        return None
    # print('making newtree.left =', tree.right)
    newtree.left = mirror(tree.right)
    newtree.right = mirror(tree.left)
    return newtree  
    
def checkmirror(tree1, tree2):
    if (tree1 == None) ^ (tree2 == None):
        return False
    if tree1 == None and tree2==None:
        return True
    if tree1.value == tree2.value:
        ans = checkmirror(tree1.left, tree2.right) and checkmirror(tree1.right, tree2.left)
        return ans
    else:
        return False

def test96(n=4):
    for i in range(n):
        test1 = bt.tree(height=4)
        test2 = mirror(test1)
        if checkmirror(test1, test2) != True:
            print('Error with', test1, test2)
            break
    test3 = bt.tree(height=8)
    if checkmirror(test3, test3):
        print('Error with', test3)
    else:
        print('Passed tests for #9.6')


"""
#9.7: Given array of ints, return index i so a[i] greater that neighbors 
""" 

def fun97(arr:list) -> int:
    l = len(arr)
    for i in range(l):
        if i == 0:
            if arr[i] > arr[i+1]:
                return i
        elif i == l-1:
            if arr[i] > arr[i-1]:
                return i
        else:
            if arr[i-1] < arr[i] and arr[i] < arr[i+1]:
                return i
    print('None (this is an error)')

# test = [random.randint(-100,100) for i in range(20)]
# fun97(test)

def fun97b(arr:list) -> int:
    s = 0
    e = len(arr)-1
    while True:
        m = (s+e)//2
        if m > 1:
            l = arr[m-1]
        else:
            l = -np.inf
        if m < len(arr) - 1:
            r = arr[m+1]
        else:
            r = -np.inf
        if l < arr[m] > r:
            return m
        elif l >= arr[m]:
            e = m-1
        else:
            s = m+1
        
# test = [random.randint(-100,100) for i in range(10)]
# p = fun97b(test)
# a = max(0,p-1)
# b = min(19,p+1)
# print(test, p, test[a:b+1])


"""
#9.8: Compute correlation 
""" 

def fun98(xs,ys):
    n = len(xs)
    meanx = sum(xs)/n
    meany = sum(ys)/n
    xs = [x - meanx for x in xs]
    ys = [y - meany for y in ys]
    
    stdx = math.sqrt(sum([x**2 for x in xs])/n)
    stdy = math.sqrt(sum([y**2 for y in ys])/n)
    
    xs = [x/stdx for x in xs]
    ys = [y/stdy for y in ys]
    
    return sum([xs[i]*ys[i] for i in range(n)])/n
    
    
"""
#9.9: Compute diameter of binary tree 
"""     


def fun99(node, dic):
    dic[node] = [0,0,0]
    for i, kid in enumerate([node.left,node.right]):
        if kid != None:
            fun99(kid,dic)
            dic[node][i] = 1 + max(dic[kid][:2])
            dic[node][2] = max(dic[node][2], dic[node][0] + dic[node][1], dic[kid][2])
    return dic
            
# tr = bt.tree(height=4)
# print(tr)    
# fun99(tr, dic={})[tr]
            
"""
#9.10: "Random" array of n ints that sum to T and within a*mean of mean
"""

def fun910(n,T,a):
    mu = T/n
    low = -int(-mu*(1-a))
    high = int(mu*(1+a))
    r = high-low
    tar_sum = T - low*n
    
    cur_sum = 0
    work = [0]*n
    while cur_sum < tar_sum:
        i = random.randint(0, n-1)
        work[i] = random.randint(0,r)
        cur_sum = sum(work)
        if cur_sum > tar_sum:
            work[i] = work[i] - (cur_sum - tar_sum)
    return [low+w for w in work]


"""
#9.11: Shortest path between given vertices in a graph
""" 

# class GraphM:
#     def __init__(self,verts:list,oEdges:dict):
#         self.verts = verts
#         self.oEdges = oEdges # a dict with keys in verts and values lists of verts


def fun911(g,s,e):
    visited = set([s])
    q = deque([[s]])
    while len(q) > 0:
        path = q.popleft()
        print(path)
        if path[-1] == e:
            return path
        for new in g.oEdges[path[-1]]:
            if new not in visited:
                visited.add(new)
                q.append(path+[new])
    print('No path from',s,'to',e)


# g = nx.fast_gnp_random_graph(20,.15)
# test = GraphM(list(g.nodes),{v:set(g.adj[v]) for v in list(g.nodes)})        
# nx.draw_kamada_kawai(g,with_labels=True)
# fun911(test, 14,10)

"""
#9.12:  For strings A and B (think len(A) >> len(B)), 
return indices of A where an anagram of B starts
"""   

def fun912(a:str,b:str)->list:
    def char_dict(c:str)->dict:
        out = {}
        for x in c:
            if x in out.keys():
                out[x] += 1
            else:
                out[x] = 1
        return out
    
    b_dict = char_dict(b)
    
    def isAnagram(dic, start, lst):
        for x in b_dict.keys():
            if dic[x] != b_dict[x]:
                return lst
        lst += [start]
        return lst
    
    index_list = []
    n = len(b)
    cur_dict = char_dict(a[:n])
    isAnagram(cur_dict, 0, index_list)    
    
    for i in range(n,len(a)):
        cur_dict[a[i-n]] -= 1
        if a[i] in cur_dict.keys():
            cur_dict[a[i]] += 1 
        else:
            cur_dict[a[i]] = 1
        isAnagram(cur_dict, i-(n-1), index_list)
        
    return index_list
             
# fun912('abchbcdbac','abc')

    
"""
#9.13: Minimal number of vertices to remove from graph so there are no edges
"""

class Graph1:
    def __init__(self,verts:set,edges:dict):
        self.verts = verts
        self.edges = edges # a dict with keys in verts and values set of verts
        self.degVert = {v:len(self.edges[v]) for v in self.verts}
        self.numEdges = sum([self.degVert[v] for v in self.edges])/2
    
    def removeVertex(self, vertex):
        edges = self.edges.pop(vertex)
        self.numEdges -= len(edges)
        for vert in edges:
            self.edges[vert].remove(vertex)
            self.degVert[vert] -= 1
        self.verts.remove(vertex)
        del self.degVert[vertex]
        return self
    
    def maxNoEdgeSubgraph(self, return_removed = True):
        removed = []
        while self.numEdges > 0:
            vert = max(self.degVert, key=self.degVert.get)
            self.removeVertex(vert)
            removed += [vert]
        if return_removed == True:
            return removed
        else:
            return self.verts



# g = nx.fast_gnp_random_graph(20,.1)
# test = Graph1(set(g.nodes),{v:set(g.adj[v]) for v in list(g.nodes)})    
# nx.draw_kamada_kawai(g,with_labels=True)
# test.maxNoEdgeSubgraph()


"""
#9.13.5: Previous problem (vertices are intervals and edges when intervals
intersect) without using language of graph theory
"""

def fun913(lst): #lst = [[1,4], [3,5],[-1,10]] 
    lst = sorted(lst)
    vert = {i:item for i,item in enumerate(lst)}
    
    comp_inter = vert[0]
    throwout = []
    keep = []
    
    for inter in lst[1:]:
        if inter[0] < comp_inter[1]: # they intersect need to throw one out
            if inter[1] <= comp_inter[1]: # inter subset comp_inter, throw out comp_inter and inter new comp
                throwout += [comp_inter]    
                comp_inter = inter
            else:
                throwout += [inter]
        else: # comp_inter[1] <= inter[0] they are disjoint, keep comp_inter, start using inter to compare 
            keep += [comp_inter]
            comp_inter = inter
    keep += [comp_inter]
    return throwout, keep


# test = [sorted([random.randint(-100,100), random.randint(-100,100)]) for i in range(100)]
# fun913(test)[1]


"""
#9.14: Parition a set of strings by the anagram equivalence relation
"""

class AnagramAndStrings:
    def __init__(self,anagram,strings):
        self.anagram = anagram
        self.strings = strings

class MyString:
    def __init__(self, string):
        self.string = string
        self.length = len(string)
        self.dic = self.make_dic(string)
        
    @staticmethod
    def make_dic(string):
        out = {}
        for x in string:
            if x in out.keys():
                out[x] += 1
            else:
                out[x] = 1
        return out

def fun914(stringList):
    build = {}
    
    for string in stringList:
        print(string)
        mystring = MyString(string)
        
        if mystring.length not in build.keys():
            build[mystring.length] = [AnagramAndStrings(mystring.dic, [mystring.string])]
        else:
            flag = True
            for item in build[mystring.length]:
                if item.anagram == mystring.dic:
                    item.strings += [mystring.string]
                    flag = False
                    break
            if flag:
                build[mystring.length] += [AnagramAndStrings(mystring.dic, [mystring.string])]
    
    return [anagram.strings for item in build.values() for anagram in item]
             
# print(fun914(['abc','abasg','cba','qvqwgf','ewr']))

"""
#9.15: number of connected components from an adjacenty matrix
Pick vertex, do dfs/bfs till exhaustion, then repeat.
"""
from scipy import sparse

def fun915(sp_mat):
    def dfs_A(sp_mat,start):
        visited = {start}
        stack = deque([start])
        while len(stack) > 0:
            here = stack.pop()
            for there in sparse.find(sp_mat[here])[1]:
                if there not in visited:
                    visited.add(there)
                    stack.append(there)
        return visited
    
    n = sp_mat.shape[0]
    verts = set(range(n))
    components = []
    
    while len(verts) > 0:
        start = verts.pop()
        component = dfs_A(sp_mat, start)
        components += [component]
        verts = verts - component
        
    return components
        
# g = nx.fast_gnp_random_graph(20,.05)
# mat = nx.adjacency_matrix(g)
# fun915(mat)


"""
#9.16: Remove kth item from end of linked list
"""

class Linked:
    def __init__(self,data,nex=None):
        self.data = data
        self.nex = nex
    
    def toList(self):
        out = []
        cur = self
        while cur != None:
            out.append(cur.data)
            cur = cur.nex
        return out
        
def toLinked(lst)->Linked:
    head = Linked(lst[0])
    cur = head
    for i in range(1,len(lst)):
        cur.nex = Linked(lst[i])
        cur = cur.nex
    return head
        

def fun916(head,k):
    lst = head.toList()
    if k == 1:
        lst = lst[:-1]
    else:
        lst = lst[:-k]+lst[-k+1:]
    return toLinked(lst)

# hey = [4,5,2,1,5,6]
# heyLink = toLinked(hey)
# heyLinkList = heyLink.toList()       
# fun916(heyLink,1).toList()    
    
    
    
"""
#9.17: Estimate pi via monte carlo
"""       

def calcPi(n=1000):
    count = 0
    for i in range(n):
        if random.random()**2 + random.random()**2 < 1:
            count += 1
    return 4*count/n


"""
#9.18: Parenthesis parser (remove minimal number of them to make string valid)
"""  

def fun918(string):
    def helper(string,wall=-1):
        count = 0
        newstring = ''
        if wall == 1:
            string = string[::-1]
        for char in string:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count == wall:
                count = 0
            else:
                newstring += char
        if wall == 1:
            newstring = newstring[::-1]
        return newstring
    
    return helper(helper(string),1)


def fun918_stack(string):
    stack = deque()
    result = ['']*len(string)
    
    for i, char in enumerate(string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if len(stack) > 0:
                result[stack.pop()] = '('
                result[i] = ')'
        else:
            result[i] = char
    
    return ''.join(result)
                
            
# fun918_stack('1wqdf))((ab)(((SD)(VFC))')
        
"""
#9.19: Generate S_n (aka give all permutation of a list of distinct ints)
"""

def fun919(n=3):
    def helper(lst):
        newlst = []
        n = len(lst[0])
        for i in range(n+1):
            for perm in lst:
                newperm = perm[:i] + [n] + perm[i:]
                newlst.append(newperm)
        return newlst
    out = [[0]]
    for i in range(n-1):
        out = helper(out)
    return out

"""
#9.20: Sample from [0,...,k-1] with weights w_0,...,w_k-1
"""    

def fun920(lst):
    cumsum = np.cumsum(lst)
    ran = random.randint(1, cumsum[-1])    
    return min([i for i in range(len(lst)) if ran <= cumsum[i]])

"""
#9.21: For two arrays of integers return the longest shared subarray
"""

def fun921(arr1,arr2):
    n, m = len(arr1), len(arr2)
    
    mat = np.zeros((n,m),dtype = int) # mat[i]
    for i, x in enumerate(arr1):
        for j, y in enumerate(arr2):
            if x == y:
                mat[i,j] = 1
    
    best = 0
    loc = [0,0]
    for off in range(-n+1,m):
        cur = 0
        for i, num in enumerate(mat.diagonal(offset=off)):
            if num == 1:
                cur += 1
                if cur > best:
                    loc = [off,i-cur+1]
                    best = cur
            if num == 0:
                cur = 0
    
    if loc[0] <= 0:
        subarr = arr2[loc[1]:loc[1]+best]
    if loc[0] >= 0:
        subarr = arr1[loc[1]:loc[1]+best]
    return best, subarr
    
    

# a = [random.randint(0,9) for i in range(random.randint(20,500))]
# b = [random.randint(0,9) for i in range(random.randint(20,500))]
# fun921(a,b)


"""
#9.22: From list of nonnegative integers return largest sum of increasing sublist
"""

def fun922(arr):
    best = arr[0]
    best_ind = [0,0]
    cur = arr[0]
    cur_ind = [0,0]
    for i in range(1,len(arr)):
        if arr[i] >= arr[i-1]:
            cur_ind[1] = i
            cur += arr[i]
            if cur > best:
                best = cur
                best_ind = cur_ind
        else:
            cur_ind = [i,i]
            cur = arr[i]
    return best, arr[best_ind[0]:best_ind[1]+1], arr

# test = [random.randint(0, 20) for i in range(20)]
# fun922(test)

"""
#9.22a: From list of nonnegative integers return largest sum of increasing sublist (indices just need to be ordered, not sequential)
Issue:
    5,3,100 best is 5+100=105
    5,3,100,10,11,95 best is 5+10+11+95=121
    If we just store best sum from previous problems, we would not see this 121 sum, since 5+10+11=26
    is not the best sum in [5,3,100,10,11].  But 26 is the best sum that must use that 11.
"""

def fun922a(arr):
    ht = [arr[i] for i in range(len(arr))] #ht[i] stores best sum that ends with arr[i] 
    best = arr[0]
    for i in range(1,len(arr)):
        ai = arr[i]
        for j in range(i):
            if ai >= arr[j]: #this condition forces us to only  
                if ht[i] < ht[j] + ai:
                    ht[i] = ht[j] + ai
                    if ht[i] > best:
                        best = ht[i]
    return best, ht
            
# hey = [5,3,100,10,11,95] + list(range(4,20))
# fun922a(hey)


"""
#9.23: For positive integer n, return k the fewest number of perfect squares needed to sum to n
Note: If n = n1 + n2, then f(n) <= f(n1) + f(n2) with equality for best split, so just minimize over n1+n2 = n
"""

def fun923(n):
    best = {1:1,2:2,3:3,4:1,5:2} # dict that stores solutions to subproblems
    for num in range(6,n+1):
        x = np.sqrt(num)
        if int(x) == x:
            best[num] = int(x)
        else:
            best[num] = min([best[i]+best[num-i] for i in range(1,int(num/2))])
    return best[n]            

"""
#9.24: Impliment n choose k and actual lists.  Can use f(n,k) = f(n-1,k-1) + f(n-1,k)
"""

def ch(n,k):
    if n<=0 or k<=0 or n < k:
        return []
    if n == k:
        return [list(range(1,n+1))]
    if k==1:
        return [[i] for i in range(1,n+1)]
    else:
        return ch(n-1,k) + [x+[n] for x in ch(n-1,k-1)]
    
    
"""
#9.25: For random string of (()(()))(, return longest well-formed substring
""" 

# If work_sum > 0 and the end of this function, the last string
# in wfss will have too many '('.  Can get around this by throwing
# it into this function again with the order reversed and swapping '(' with ')'  
# Or just run the function twice with pstr and its mirror with run in O('2'n)

def parPar(pstr):
    val = {'(':1,')':-1}
    wfss = []
    work_sum = 0
    work_str = ''
    for p in pstr:
        # print('working on',work_str,'new char',p)
        work_sum += val[p]
        if work_sum == -1:
            if len(work_str) > 0:
                wfss.append(work_str)
            work_str = ''
            work_sum = 0
        else:
            work_str += p
    if len(work_str) > 0:
        wfss.append(work_str)
    return wfss

# def mirror(string):
#     out = ''
#     swap = {'(':')',')':'('}
#     for p in string[::-1]:
#         out += swap[p]
#     return out

# This solution only needs a single pass so it is O(n)    
def stackPar(pstr):
    wfss = []
    work = []
    stack = deque()
    start = 0
    for i,p in enumerate(pstr):
        if p == '(':
            work += [''] 
            stack.append(i-start)
        elif p == ')':
            if stack:
                work[stack.pop()] = '('
                work+= [')']
            else:
                if len(work) > 0:
                    wfss.append(''.join(work))
                start = i + 1
                work = []
    final = ''
    for c in work:
        if c == '':
            if len(final) > 0:
                wfss.append(final)
                final=''
        else:
            final = final + c
    if len(final) > 0: wfss.append(final)     
    return wfss
                

def ranParStr(n=20):
    out = ''
    for i in range(n):
        x = random.randint(0, 1)
        if x == 0:
            out += '('
        else:
            out += ')'
    return out

# parPar(ranParStr())

# test = ranParStr(30)
# print(test, stackPar(test))


def cleanPar(pstr):
    stack = deque([-1]) # stack[-1] is either the most recent unmatched '(' or the last illegal ')'
    best = 0
    string = ''
    for i,p in enumerate(pstr):
        if p == '(':
            stack.append(i)
        else:
            stack.pop()  # this append/pop is the standard +/- 1 
            if len(stack) == 0: # since len(stack)=1 at start, this means we hit an illegal ')'
                stack.append(i) 
            else: 
                if i-stack[-1] > best:
                    best = i-stack[-1]
                    string = pstr[stack[-1]+1: i+1]
    return best, string

"""
#9.26: For matrix with positive integers find longest path of increasing numbers (diagonal counts as neighbor).
This is the problem: Find the diameter of a directed acyclic graph 
"""

def generate_random_dag(n, p):
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True)
    random_dag = nx.DiGraph(
        [
            (u, v) for (u, v) in random_graph.edges() if u < v
        ]
    )
    return random_dag

class GraphM:
    def __init__(self,verts:list,oEdges:dict):
        self.verts = verts
        self.oEdges = oEdges # a dict with keys in verts and values lists of verts
        self.iEdges = self.inEdges(self.verts, self.oEdges)
    
    @staticmethod
    def inEdges(verts, oEdges):
        dic = {v:set() for v in verts}
        for v in verts:
            for w in oEdges[v]:
                dic[w].add(v)
        return dic
    
    def topSort(self) -> list:
        # self.compIedges()
        ideg = {v:len(self.iEdges[v]) for v in self.verts}
        queue = deque([v for v in self.verts if ideg[v]==0])
        toplist = []
        
        while len(queue) > 0:
            v = queue.popleft()
            toplist.append(v)
            for w in self.oEdges[v]:
                ideg[w] -= 1
                if ideg[w] == 0:
                    queue.append(w)    
        return toplist



def fun926(gr):
    best = 0
    q = deque([[0,v] for v in gr.verts])
    while q:
        n, v = q.popleft()
        if not gr.oEdges[v]:
            if n > best:
                best = n
        else:
            for w in gr.oEdges[v]:
                q.append([n+1,w])
    return best
# Run time is terrible here.  Can be improved by computing a topological sort first



# This runs is O(n) time
def helpTopSort(gr):
    inDeg = {v:len(gr.iEdges[v]) for v in gr.verts}
    qu = deque([v for v in gr.verts if inDeg[v]==0])
    toplist = []
    while qu:
        v = qu.popleft()
        toplist.append(v)
        for w in gr.oEdges[v]:
            inDeg[w] -= 1
            if inDeg[w] == 0:
                qu.append(w)
    return toplist
        
# This runs in O(n^2) time
def fun926wTop(gr):
    top = helpTopSort(gr) #list of edges is a topological order
    n = len(top)
    best = [0]*n # this stores the length of the longest path that end at this vertex
    for i in range(n):
        for j in range(i):
            if top[i] in gr.oEdges[top[j]]:
                best[i] = max(best[i], best[j]+1)
    return max(best)
        
    
def test926(n=20,p=.4,pic=False):
    g = generate_random_dag(n, p)
    actual = nx.dag_longest_path_length(g)
    test = GraphM(list(g.nodes),{v:set(g.adj[v]) for v in list(g.nodes)})
    # testans = fun926(test)
    testans = fun926wTop(test)
    if pic:
        nx.draw_kamada_kawai(g, arrows=True, with_labels=True)
    print(actual, '=', testans, 'is', actual==testans)


def fun926M(mat):
    def nbhd(loc,n,m):
        alldirec = [np.array([ud,rl]) for ud in [-1,0,1] for rl in [-1,0,1]]
        allpos = [np.array(loc) + x for x in alldirec] 
        return [tuple(x) for x in allpos if (x[0] in range(n) and x[1] in range(m))]
    
    n,m = mat.shape
    vert = [(i,j) for i in range(n) for j in range(m)]
    edgeDic = {v:set() for v in vert}
    for v in vert:
        for w in nbhd(v,n,m):
            if mat[v] < mat[w]:
                edgeDic[v].add(w)
    
    return GraphM(vert,edgeDic)

# fun926M(hey).verts
# fun926M(hey).oEdges

"""
#9.27: For given n, find the number of ways to write n as a sum of consecutive positive integers
Doing it in O(n)
"""

def fun927(n):
    def cSum(s,k):
        highsum = (s+1+k)*(s+k)/2 # sum of 1+2+3+...+(s+k)
        lowsum = (s+1)*s/2 # sum of 1+2+3+..+s
        return highsum-lowsum #sum of (s+1)+...+(s+k)
    
    def bound(s,m): # if cSum(s,k) = m, then k is within [0,3] of bound(s,n)
        return max(0,int(math.sqrt(2*n+s**2))-s - 2)

    out = []
    for s in range(n+1):
        b = bound(s,n)
        for k in range(b,b+4):
            if cSum(s,k) == n:
                out.append([s,k])
    return out

"""
#9.28: Class based solution to having a live median computation from stream of numbers
"""

def med(heapMax, heapMin):
    if len(heapMax) == 0 and len(heapMin)== 0:
        return 0
    if len(heapMax) > len(heapMin):
        out = -heapMax[0]
    elif len(heapMin) > len(heapMax):
        out = heapMin[0]
    else:
        out = (-heapMax[0]+heapMin[0])/2
    return out
        

class NumStream:
    def __init__(self, stream, maxH=[],minH = []):
        self.stream = deque(stream)
        self.maxH = maxH
        self.minH = minH
        self.median = med(self.maxH, self.minH)
    
    def update(self):
        
        new = self.stream.popleft()
        # print(new)
        if new >= self.median:
            heapq.heappush(self.minH, new)
        else:
            heapq.heappush(self.maxH, -new)
        
        n = len(self.maxH) - len(self.minH)
        if n == -2:
            heapq.heappush(self.maxH, -heapq.heappop(self.minH))
        elif n == 2:
            heapq.heappush(self.minH, -heapq.heappop(self.maxH))
        
        self = NumStream(self.stream,self.maxH,self.minH)
        return self.median
    
    def medList(self):
        out = []
        while self.stream:
            # print(self.stream, self.maxH, self.minH)
            newmed = self.update()
            
            out.append(newmed)
        return out
    
    
# test = [random.randint(-100, 100) for i in range(20)]
# hey = NumStream(test)
# medians = hey.medList()
# realmed = [np.median(test[0:i]) for i in range(1,21)]         
        

"""
#9.30: Do gradient descent
Have points p_1,...,p_n in R^2
find point x in R^2 that minimizes sum of distances to the points
"""

def dist(x,y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2) 

def gradD(lst):
    def part0(x):
        return sum([(x[0]-p[0])/dist(x,p) for p in lst])
    def part1(x):
        return sum([(x[1]-p[1])/dist(x,p) for p in lst])
    
    rate = 1
    decay = 1e-3
    stopRate = 1e-7
    damp = 0
    
    x_w = [0,0]
    d0 = 0
    d1 = 0
    
    while rate > stopRate:
        d0 = part0(x_w) + damp*d0
        d1 = part1(x_w) + damp*d1
        
        x_w[0] = x_w[0] - rate*d0
        x_w[1] = x_w[1] - rate*d1
        
        rate = rate*(1-decay)
        
    return x_w
    
test = [[random.randint(-10, 10), random.randint(-10, 10)] for i in range(10)]    

gradD(test)    