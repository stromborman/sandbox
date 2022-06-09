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


"""
Level order traversal (aka BFS):
    Output is a list of list, where the ith list is
    the value of the nodes in level i (ordered left to right)
"""

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    if root==None:
        return []
    
    out = []
    cur = -1
    
    levels = {root:0}
    q = deque([root])
    
    
    while q:
        node = q.popleft()
        if levels[node] > cur:
            cur += 1
            out.append([node.val])
        else:
            out[cur].append(node.val)
        
        if node.left:
            levels[node.left] = levels[node] + 1
            q.append(node.left)
        if node.right:
            levels[node.right] = levels[node] + 1
            q.append(node.right)
            
    return out


"""
Right side view:
    Return a list of the right most node values in each level
"""

def rightSideView(root: Optional[TreeNode]) -> List[int]:
    if root==None:
        return []

    out = []
    cur = -1

    levels = {root:0}
    q = deque([root])


    while q:
        node = q.popleft()
        if levels[node] > cur:
            cur += 1
            out.append([node.val])
        else:
            out[cur].append(node.val)

        if node.left:
            levels[node.left] = levels[node] + 1
            q.append(node.left)
        if node.right:
            levels[node.right] = levels[node] + 1
            q.append(node.right)

    return [item[-1] for item in out]


"""
Count Good Nodes in Binary Tree:
    Given a binary tree root, a node X in the tree is named good 
    if X.val is the max value in the path from root to X.
    Return the number of good nodes in the binary tree.
"""

def goodNodes(root: TreeNode) -> int:
    
    count = 1
    q = deque([(root,root.val)])
    
    while q:
        node, maxval = q.popleft()
        
        for kid in [node.left,node.right]:
            if kid:
                q.append((kid,max(maxval,kid.val)))
                if kid.val >= maxval:
                    count += 1
                    
    return count


"""
Verify BST
"""

def isValidBST(root: Optional[TreeNode]) -> bool:
    
    def helper(low,node,high):
        if node == None:
            return True
        if node.val <= low or node.val >= high:
            return False
        return helper(low, node.left, node.val) and helper(node.val, node.right, high)
    
    return helper(float('-inf'),root,float('inf'))



"""
k-th smallest element in BST
"""

def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right


def kthSmallestRecurvise(root: Optional[TreeNode], k: int) -> int:
    
    def leftHereRight(node,lst,k):
        if node == None:
            return
        if len(lst) < k:
            leftHereRight(node.left,lst,k)
        if len(lst) < k:
            lst.append(node.val)
        if len(lst) < k:
            leftHereRight(node.right,lst,k)
        return lst
    
    return leftHereRight(root,[],k)[-1]



"""
Build tree from Preorder and Inorder list of values:
    Assume that values in tree are unique
"""



def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    root = TreeNode(preorder[0])
    q = deque([(root, preorder, inorder)])
    while q:
        node, preorder, inorder = q.popleft()
        
        # This causes the TC to be multiplied by O(n)
        # Can be avoided by making a hashmap for inorder val->index, ie
        # {val: index for index,val in enumerate(inorder)}
        split = inorder.index(node.val)
        
        # Due to the slicing below, we lose track of the global indicies
        # for the inorder list.  So if we wanted to use a hash map
        # this needs to be rewritten without slicing
        if split != 0:
            node.left = TreeNode(preorder[1])
            q.append((node.left, preorder[1:split+1], inorder[:split]))
        
        if split < len(preorder) - 1:
            node.right = TreeNode(preorder[split+1])
            q.append((node.right,preorder[split+1:], inorder[split+1:]))
    
    return root





