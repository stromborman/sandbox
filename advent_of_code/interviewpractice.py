#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt


"""
#1 Sorted list of dates, return pairs of dates one month apart
"""

import datetime

# this is ill-defined, e.g. what is one month after 1/31? (last day of feb?)

def f1(lst:list) -> list:
    d = datetime.timedelta(days=30)
    out = []
    for i in range(len(lst)-1):
        if lst[i+1] - lst[i] == d:
            out.append((lst[i],lst[i+1])) 
    return out

# f1([datetime.date(2000,1,1), datetime.date(2000,1,31), datetime.date(2000,2,27), datetime.date(2000, 3,29)])


"""
#2 Array of strings, return first string that only appears once
"""

def f2(lst:list) -> str:
    dic = {}
    for string in lst:
        if string in dic.keys():
            dic[string] += 1
        else:
            dic[string] = 1
    out = None
    for string in lst:
        if dic[string] == 1:
            out = string
            break
    return out

# f2(['as','qwer','as','sefv'])

"""
#3 Implement simplified count vectorizer with fit and transform functionality.
"""

corp1 = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

from scipy.sparse import lil_matrix

class SCV:
    def __init__(self):
        self.words = None
        self.counts = None
        
    def fit(self, corpus):
        self.words = set()
        for part in corpus:
            clean_part = [ word.lower().strip('.!?,:;') for word in part.split(' ')]
            self.words = self.words.union(set(clean_part))
        self.words = list(self.words)
        self.word2num = {x:i for i,x in enumerate(self.words)}
        self.num2word = {i:x for i,x in enumerate(self.words)}
        
            
    def transform(self, corpus):
        n = len(self.words)
        l = len(corpus)
        lil = lil_matrix((l, n), dtype=int)
        for i, part in enumerate(corpus):
            for cword in [ word.lower().strip('.!?,:;') for word in part.split(' ')]:
                lil[i,self.word2num[cword]] += 1
        self.counts = lil
        
        
# test = SCV()
# test.fit(corp1)
# test.transform(corp1)
# test.counts.toarray()  
           
"""
#4 Sorted lists, merge them and return as 1 single sorted list. 
"""    

from collections import deque        
        
def f4(l1:list,l2:list)->list:
    q1 = deque(l1)
    q2 = deque(l2)
    out = []
    n = len(q1) + len(q2)
    
    for i in range(n):
        if len(q1) == 0:
            out.append(q2.popleft())
        elif len(q2) == 0:
            out.append(q1.popleft())
        elif q1[0] <= q2[0]:
            out.append(q1.popleft())
        else:
            out.append(q2.popleft())
    
    return out
    
        
# x1 = [1,5,5,23,55]
# x2 = [-1, 0, 5, 10, 23, 56]    
# f4(x1,x2)


"""
#5 Corner points (from list of tuples return points that have nothing smaller elementwise) 
"""  

def f5(lst:list)->list:
    lst.sort(key= lambda t: t[0])
    corners = []
    y_min = np.inf
    for p in lst:
        if p[1] <= y_min:
            corners.append(p)
            y_min = p[1]
    return corners
        
    
def test5(n=20):
    test = []    
    for i in range(n):    
        test.append([random.random(),random.random()])
    corners = np.transpose(f5(test))
    plt.scatter(np.transpose(test)[0],np.transpose(test)[1])
    plt.scatter(corners[0], corners[1], marker='x')

"""
#6 Topological sort: From directed acyclic graph, list vertices without breaking the graph order
"""  

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
        
import networkx as nx    


def generate_random_dag(n, p):
    random_graph = nx.fast_gnp_random_graph(n, p, directed=True)
    random_dag = nx.DiGraph(
        [
            (u, v) for (u, v) in random_graph.edges() if u < v
        ]
    )
    return random_dag


def checkTopSort(lst:list,gr:GraphM) -> bool:
    order = {item:i for i,item in enumerate(lst)}
    out = True
    for v in gr.verts:
        if out == False: break
        for w in gr.oEdges[v]:
            if order[v] > order[w]:
                out = False
                break
    return out


def test6(n=20,p=.4,pic=False):
    g = generate_random_dag(n, p)
    test = GraphM(list(g.nodes),{v:set(g.adj[v]) for v in list(g.nodes)})
    hey = test.topSort()
    if pic:
        nx.draw_kamada_kawai(g, arrows=True, with_labels=True)
    print(checkTopSort(hey, test))
    
# test6(pic=True)
    

"""
#7 Binary search of sorted list
"""             

def binSearch(lst:list,t:int) -> int:
    # Returns index i such that lst[i] = t else largest i such that lst[i] < t
    left = 0
    right = len(lst)-1
    while left <= right and right !=0:
        mid = left + (right-left)//2
        if lst[mid] == t:
            return mid
        elif lst[mid] < t:
            left = mid+1
        else:
            right = mid-1
    return right

def test7(n=30):
    tests = []
    for i in range(10):
        lst_t= sorted([random.randint(-n, n) for i in range(n//2)])
        t_t = random.randint(-int(1.5*n), int(1.5*n))
        tests.append([lst_t,t_t])
    
    for testo in tests:
        j = binSearch(testo[0], testo[1])
        if testo[0][j]<testo[1]:
            if j < len(testo[0])-1:
                if testo[0][j+1] <= testo[1]:
                    print('Error1')
                    return testo[0],testo[1],testo[0][j]
        elif testo[0][j] > testo[1] and j!=0:
            print('Error2')
            return testo[0],testo[1],testo[0][j]
    print('Passed')
    
"""
#7.5 From a sorted list, find k elements closest to target
"""      

def findClosest(lst:list,k:int,target:int) -> list:
    i = binSearch(lst, target)
    if lst[i] != target:
        if i != len(lst)-1:
            if abs(lst[i+1]-target)<abs(lst[i]-target):
                i = i+1
    left = i
    right = i+1
    
    while right - left < k:
        # print(left,right,len(lst))
        if left == 0 and right == len(lst):
            break
        elif left == 0:
            right = right + 1
        elif right == len(lst):
            left = left - 1
        else:
            if abs(lst[left - 1] - target) <= abs(lst[right]-target):
                left = left -1
            else:
                right = right + 1
    return lst[left:right]

# n=30
# tests = []
# for i in range(10):
#     lst_t= sorted([random.randint(-n, n) for i in range(n//2)])
#     k_t = random.randint(1,n//2)
#     t_t = random.randint(-int(1.5*n), int(1.5*n))
#     tests.append([lst_t,k_t,t_t])
    
# for item in tests:
#     print(findClosest(*item))
#     print(*item)

"""
#8 Search of binary tree for node
"""  
import binarytree as bt

# test = bt.tree(height=4)
# print(test)


def dfsList(root, lst): #Preorder
    if root == None:
        return
    lst.append(root.value)
    dfsList(root.left,lst)
    dfsList(root.right,lst)    
    return lst

# dfsList(test,[])

# def dfsInP(root):
#     if root == None:
#         return
#     dfsInP(root.left)
#     print(root.value)
#     dfsInP(root.right)

# dfsInP(test)

def dfsIn(root,lst): #Inorder
    if root == None:
        return
    dfsIn(root.left,lst)
    lst.append(root.value)
    dfsIn(root.right,lst)
    return lst

# print(dfsIn(test,[]))


"""
#8.5 BFS for trees
"""  

def bfsList(root):
    lst = []
    q = deque()
    q.append(root)
    while len(q) > 0:
        vert = q.popleft()
        lst.append(vert.value)
        if vert.left:
            q.append(vert.left)
        if vert.right:
            q.append(vert.right)
    return lst

# print(bfsList(test[0]))


# want the list to 'zigzag' across levels (instead of always left to right)
def bfsZig(root):
    lst = []
    q = deque()
    q.append(root)
    dic = {root:0}
    level = 0
    cur_list = []
    
    while len(q) > 0:
        work = q.popleft()
        
        if dic[work] > level:
            level = dic[work]
            if level % 2 == 1:
                cur_list = cur_list[::-1]
            lst = lst + cur_list
            cur_list = []
            
        cur_list.append(work)
        if work.left:
            dic[work.left] = dic[work]+1
            q.append(work.left)
        if work.right:
            dic[work.right] = dic[work]+1
            q.append(work.right)
    return [x.value for x in lst]
        
    
# test = bt.tree(height=4)
# print(test)

# bfsZig(test)


"""
#9 Binary search tree
"""  

def binSearchTree(node,target,last):
    if node == None:
        print(target,'not in tree, but it does have', last.value)
    elif node.value == target:
        print('tree has value', node.value, '=', target)
    elif node.value > target:
        binSearchTree(node.left, target, node)
    elif node.value < target:
        binSearchTree(node.right, target, node)
        

# bst_test = bt.bst(height=5)
# print(bst_test)

# binSearchTree(bst_test, 19, None)




"""
#10 Search of binary tree: path from leaf to node
"""      


# test = bt.tree(height=3, is_perfect=True)
# print(test)print(test)

def pathDFS(target, source):
    if source == None:
        return []
    if target == source:
        return [source]
    for res in [pathDFS(target, source.left), pathDFS(target, source.right)]:
        if res:
            return [source] + res
    return []

# print(test[0], str(test[0].value)+' -> '+str(test[12].value), pathDFS(test[12], test[0]))


def pathDFS2(target, source, path):
    if target == source:
        path.append(source)
        return path # we are where we want to be
    for kid in [source.left,source.right]:
        if kid: # if the kid is not none
            path.append(source) # add the parent to the path
            temp = pathDFS2(target, kid, path) # search now from kid
            if temp:
                return temp
            path.pop() # reach here if no path from kid to target, so backtrack
    return [] # reach here if currently at leaf that is not target

# print(test[0], str(test[0].value)+' -> '+str(test[12].value), pathDFS2(test[12], test[0],[]))

def pathBFS(target, source):
    q = deque()
    flag = True
    q.append([source])
    while len(q) > 0 and flag:
        path = q.popleft()
        if path[-1] == target:
            flag = False
        else:
            for kid in [path[-1].left,path[-1].right]:
                if kid:
                    new_path = path + [kid]
                    q.append(new_path)
    return path

# print(test[0], str(test[0].value)+' -> '+str(test[12].value), pathBFS(test[12], test[0]))


"""
#11 k-th largest element from list of numbers 
"""  

def helperK(lst,x,k):
    if lst == []:
        return [x]
    else:
        flag = True
        i = 0
        while flag:
            if x >= lst[i]:
                lst = lst[0:i] + [x] + lst[i:]
                flag = False
            elif i == len(lst)-1:
                lst = lst + [x]
                flag = False
            else:
                i = i+1
    return lst[:k]
    

def findKth(lst,k):
    working = []
    for x in lst:
        working = helperK(working, x, k)
    return working[k-1]
    
def test11(n=20):
    for i in range(10):
        lst_t = [random.randint(-100,100) for i in range(n)]
        k_t = random.randint(1,n//2)
        myk = findKth(lst_t, k_t)
        realk = sorted(lst_t, reverse = True)[k_t-1]
        if myk != realk:
            print('Failure for', lst_t, k_t)
            print('Got', myk, 'should have got', realk)
            break
    print('Passed tests for #11')
        
"""
#11.1 k-th largest element from list of numbers (with heaps) 
"""  

import heapq

hey = [random.randint(-100,100) for i in range(20)]
heapq.nlargest(4, hey) == sorted(hey,reverse=True)[:4]


"""
#12 Live updating of median of stream using heaps 
"""  


def streaMed(a):
    med_list = [a[0],np.median(a[:2])]
    left_max = [-min(a[:2])]
    len_left = 1
    right_min = [max(a[:2])]
    len_right = 1
    for new in a[2:]:
        if new <= med_list[-1]:
            heapq.heappush(left_max,-new)
            len_left += 1
        else:
            heapq.heappush(right_min,new)
            len_right += 1
        
        if len_left == len_right - 2:
            x = heapq.heappop(right_min)
            heapq.heappush(left_max,-x)
            len_left += 1
            len_right -= 1
        elif len(right_min) == len(left_max) - 2:
            x = heapq.heappop(left_max)
            heapq.heappush(right_min,-x)
            len_left -= 1
            len_right += 1
        
        b = len_left - len_right
        if b==1:
            new_med = -left_max[0]
        if b==0:
            new_med = (-left_max[0]+right_min[0])/2
        if b==-1:
            new_med = right_min[0]
        med_list.append(new_med)
    return med_list



def test12(n=20):
    test =  [random.randint(-10*n,10*n) for i in range(100*n)]
    for i in range(2,1000):
        if streaMed(test)[i] != np.median(test[:i+1]):
            print('ERROR')
            break
    print('Passed test for #12')


              
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
#9.5: for arr, return max(0, sum(arr[i:j])) over all j 
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
    
    return [anagram.strings for item in hey.values() for anagram in item]
             
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
            for there in sparse.find(mat[here])[1]:
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
        
g = nx.fast_gnp_random_graph(20,.05)
mat = nx.adjacency_matrix(g)

fun915(mat)



