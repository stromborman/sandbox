#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 18
https://adventofcode.com/2021/day/18
"""
import math
import copy
from itertools import combinations

def read(filename):
    with open(filename) as file:
        lines = file.read().split('\n')
    return lines

real = read('input18')
test = read('input18_test')

class Node:
    def __init__(self,value='',left=None,right=None,depth=0,par=None):
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.par = par
        
def s2t(s):
    root = Node(depth=-1)
    node = root
    for c in s:
        if c=='[':
            node.left = Node(par=node, depth=node.depth+1)
            node=node.left
        elif c==',':
            if type(node.value) != int and node.value !='':
                node.value = int(node.value)
            node.par.right = Node(par=node.par, depth = node.depth)
            node = node.par.right
        elif c==']':
            if type(node.value) != int and node.value !='':
                node.value = int(node.value)
            node=node.par
        else:
            node.value += c
        # print(c,node.value,node.depth) 
    assert root == node, 'WARNING'    
    return root

def t2s(t):
    if t==None:
        return ''
    elif type(t.value) == int:
        return str(t.value)
    else:
        return '['+t2s(t.left)+','+t2s(t.right)+']'
            
            
def inorder(root, lst):
    if root == None:
        return
    inorder(root.left, lst)    
    if root.value != '':
        lst.append(root)
    inorder(root.right, lst)
    return lst 

    
def process(tree1):
    tree = copy.deepcopy(tree1)
    lst = inorder(tree,[])
    
    flag1 = True
    i = 0
    while flag1 and i <= len(lst)-2:
        # print('i is', i)
        if lst[i].depth >= 4:
            par = lst[i].par
            # print('par of', t2s(lst[i]), 'is', t2s(par))
            flag1=False
            if i >= 1:
                # print(par.left.value, 'is added to', lst[i-1].value)
                lst[i-1].value += par.left.value
            if i <= len(lst)-3:
                # print(par.right.value, 'is added to', lst[i+2].value)
                lst[i+2].value += par.right.value
            # print(t2s(par), 'is set to 0')
            par.value = 0
            par.left = None
            par.right = None
        i += 1
    
    flag2 = True
    j = 0
    while flag1 and flag2 and j < len(lst):
        # print('j is', j)
        node = lst[j]
        if node.value >= 10:
            val = node.value
            # print('node', t2s(node), 'gets kids')
            flag2 = False
            node.left = Node(value=math.floor(val/2),depth=node.depth+1,par=node)
            node.right = Node(value=math.ceil(val/2),depth=node.depth+1,par=node)
            node.value = ''
            # print(t2s(node))
        j += 1
    
    return tree



# sim = ['[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]', '[[[[0,7],4],[7,[[8,4],9]]],[1,1]]',
#        '[[[[0,7],4],[15,[0,13]]],[1,1]]', '[[[[0,7],4],[[7,8],[0,13]]],[1,1]]',
#        '[[[[0,7],4],[[7,8],[0,[6,7]]]],[1,1]]', '[[[[0,7],4],[[7,8],[6,0]]],[8,1]]']


# for i in range(4):
#     print(t2s(process(s2t(sim[i])))==sim[i+1])




def sim(tree):
    flag = True
    while flag:
        tree1 = process(tree)
        if t2s(tree) == t2s(tree1):
            flag = False
        else:
            tree = tree1
    return tree

def add(t,s):
    return sim(s2t('['+t2s(t)+','+s+']'))
    

def addsim(lst):
    tree = s2t(lst[0])
    for i in range(1,len(lst)):
        tree = add(tree,lst[i])
    return tree
    
    

def mag(t):
    if t==None:
        return 0
    elif type(t.value) == int:
        return t.value
    else:
        return 3*mag(t.left)+2*mag(t.right)
    
    
mag(addsim(test)) == 4140

print('Answer to part1:', mag(addsim(real)))


def adds(t,s):
    return mag(sim(s2t('['+t+','+s+']')))

allsums = list(combinations(real, 2)) + [(i,j) for (j,i) in combinations(real, 2)]


print('Answer to part2:', max([adds(i,j) for (i,j) in allsums]))    