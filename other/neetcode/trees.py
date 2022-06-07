#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Trees
"""

from collections import defaultdict, deque
from typing import List, Optional
from copy import deepcopy

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
        
"""
Invert Binary Tree
"""

def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if root == None:
        return None
    copy_right = deepcopy(root.right)
    root.right = invertTree(root.left)
    root.left = invertTree(copy_right)
    return root

"""
Maximum Depth of Binary Tree
"""

def maxDepth(root: Optional[TreeNode]) -> int:
    if root == None:
        return 0
    return max(maxDepth(root.left),maxDepth(root.right))+1

"""
Diameter of Binary Tree
"""

def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    
    def recur(node):
        if node == None:
            return {'left':-1,'right':-1,'diam':0}
        
        leftD =  recur(node.left)
        depthL = max(leftD['left'],leftD['right'])
        diamL = leftD['diam']
        
        rightD = recur(node.right)
        depthR = max(rightD['left'],rightD['right'])
        diamR = rightD['diam']
        
        return {'left':1+depthL,
                'right': 1+depthR,
                'diam': max(diamL, diamR, 2+depthL+depthR)}
    
    return recur(root)['diam']
    
    
"""
Balanced Binary Tree:
    Given a binary tree, determine if it is height-balanced.
    IE if the left and right subtrees of every node 
        differ in height/depth by no more than 1.
"""    


def isBalanced2(root: Optional[TreeNode]) -> bool:
    
    def depth(node):
    
        if node==None:
            return (True,-1)
        
        bool_left, depth_left = depth(node.left)
        if not bool_left:
            return (False,None)
        
        bool_right, depth_right = depth(node.right)
        if not bool_right:
            return (False,None)
        
        if abs(depth_left-depth_right)<=1:
            return (True, max(depth_left,depth_right)+1)
        
        else:
            return (False, None)
    
    return depth(root)[0]
  

"""
Equality for Binary Trees
"""

def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if (p == None) ^ (q == None):
        return False
    if (p == None) and (q == None):
        return True    
    return p.val==q.val and isSameTree(p.left,q.left) and isSameTree(p.right,q.right)

"""
Subtree of Another Tree
"""

def isSubtree(root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:   
    
    def dfsTraverse(tree,lst):
        if tree == None:
            return
        lst.append(tree)
        dfsTraverse(tree.right,lst)
        dfsTraverse(tree.left,lst)
        return lst

    for tree in dfsTraverse(root,[]):
        if isSameTree(tree,subRoot):
            return True
    return False

"""
Lowest (closest to root) Common Ancestor of a Binary Search Tree
"""

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    def pathDFS(target, source):
        if source == None:
            return []
        if target == source:
            return [source]
        for res in [pathDFS(target, source.left), pathDFS(target, source.right)]:
            if res:
                return [source] + res
        return []
    
    ppath = pathDFS(p,root)
    qpath = pathDFS(q,root)
    
    ans = ppath[0]
      
    for i, node in enumerate(ppath):
        try:
            if qpath[i] == node:
                ans = node
            else:
                break
        except:
            break
    return ans