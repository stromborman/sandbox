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
# print(test)

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
#13: Given a binary tree, determine if there is a path from the root to a leaf
whose values sum to a given target.  What if it can happen anywhere, just not at leaf?
"""

def sumDFS_I(node,target,end='leaf'):
    stack = [[node,0]]
    flag = False
    while stack:
        work, cur_sum = stack.pop()
        cur_sum += work.value
        
        if cur_sum == target:
            if end == 'leaf':
                if work.left == None and work.right == None:
                    flag = True
            if end == 'any':
                flag = True
            
        if work.right:
            stack.append([work.right, cur_sum])
        if work.left:
            stack.append([work.left, cur_sum])
        if work.left == None and work.right == None:
            if cur_sum == target:
                flag = True
    return flag


test = bt.tree(height=3, is_perfect=True)
print(test)
root = bt.Node(1)   
root.left = bt.Node(2)

sumDFS_I(root, 1, 'ianyi')
sumDFS_I(test,200) 


# def preorderDFS_R(node,lst):
#     if node != None:
#         lst.append(node)
#         preorderDFS_R(node.left, lst)
#         preorderDFS_R(node.right, lst)
#     return lst

# print(preorderDFS_R(test))

# def preorderDFS_I(node):
#     lst = []
#     stack = [node]
#     while stack:
#         work = stack.pop()
#         lst.append(work)
#         if work.right:
#             stack.append(work.right)
#         if work.left:
#             stack.append(work.left)
#     return lst

# print(preorderDFS_I(test))
