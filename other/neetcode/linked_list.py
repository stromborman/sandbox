#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://neetcode.io/
Linked List
"""

from collections import defaultdict, deque
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
    def __repr__(self):
        out = []
        head = self
        while head:
            out.append(head.val)
            head = head.next
        return str(out)
        
def makeLinkedList(lst):
    if lst == None:
        return None
    
    head = ListNode(lst[0])
    working = head
    for val in lst[1:]:
        working.next = ListNode(val)
        working = working.next
    
    return head

test = makeLinkedList([1,5,6,4,3,2])


"""
Reverse Linked List:
    Given the head of a singly linked list, reverse the list, 
    and return the reversed list.
"""


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    
    if head == None:
        return None
    
    stack = []
    current = head
    while current:
        stack.append(current.val)
        current = current.next
    
    newhead = ListNode(stack.pop())
    current = newhead
    while stack:
        new = ListNode(stack.pop())
        current.next = new
        current = new
    
    return newhead

"""
Merge Two Sorted Lists:
    You are given the heads of two sorted linked lists list1 and list2, which
    are sorted. Merge the two lists in a one sorted list. 
    The list should be made by splicing together the nodes of the first two lists.
"""

def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if list1 and list2:
        if list1.val <= list2.val:
            start = ListNode(list1.val)
            list1 = list1.next
        else: 
            start = ListNode(list2.val)
            list2 = list2.next

        cur = start

        while list1 or list2:
            if list1 == None:
                cur.next = list2
                break
            if list2 == None:
                cur.next = list1
                break

            if list1.val <= list2.val:
                cur.next = ListNode(list1.val)
                list1 = list1.next
            else: 
                cur.next = ListNode(list2.val)
                list2 = list2.next
            cur = cur.next
    
    elif list1:
        start = list1
        
    elif list2:
        start = list2
        
    else:
        start = None

    return start

"""
Reorder List:
    You are given the head of a singly linked-list. 

        L_0 → L_1 → ... → L_{n-1} → L_n

    Reorder the list to be on the following form:

        L0 → Ln → L1 → L_{n-1} → L_2 → L_{n-2} → ...

    You may not modify the values in the list's nodes.
"""

def reorderList(head: Optional[ListNode]) -> None:
    queue = deque()
    current = head
    while current:
        queue.append(current)
        current = current.next

    n = len(queue)
    current = queue.popleft()
    for i in range(1,n):
        if i%2 == 0:
            current.next = ListNode(queue.popleft().val)
        else:
            current.next = ListNode(queue.pop().val)
        current = current.next

"""
Remove nth Node From End of List:
    Given the head of a linked list, remove the nth node from the end of the 
    list and return its head. IE n+1 from end now connects to n-1 from end.
"""

def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if head.next == None:
            return None
        
        stack = [head]
        while stack[-1].next:
            stack.append(stack[-1].next)
        
        if n == len(stack):
            return head.next
        
        if n == 1:
            target = None
        else:
            target = stack[-n+1]

        stack[-n-1].next = target

        return head

"""
Copy List with Random Pointer:
    A linked list of length n is given such that each node contains an 
    additional random pointer, which could point to any node in the list, or null.
    Construct a deep copy of the list.
"""

class NodeR:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def copyRandomList(head: 'Optional[NodeR]') -> 'Optional[NodeR]':
    
    if head == None:
        return None
    
    def toDictAndList(head):
        dic = {}
        i = 0
        while head:
            dic[head] = i
            i= i+1
            head=head.next
        dic[None] = i
        lst = [node for node in dic.keys()]
        return dic, lst
    
    dic, lst = toDictAndList(head)
    
    head_copy = NodeR(head.val)
    lst_copy = [head_copy]
    work = head_copy
    for node in lst[1:-1]:
        new = NodeR(node.val)
        lst_copy.append(new)
        work.next = new
        work = new
    lst_copy.append(None)
    
    for i,node in enumerate(lst_copy[:-1]):
        node.random = lst_copy[dic[lst[i].random]]
        
    return head_copy



"""
Add Two Numbers:
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in reverse order, and each of their nodes contains a single digit. 
    Add the two numbers and return the sum as a linked list.
"""

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    
    def getNum(linked):
        num = ''
        while linked:
            num = num + str(linked.val)
            linked = linked.next
        return int(num[::-1])
    
    def getLinked(num):
        lst = list(str(num))
        head = ListNode(int(lst.pop()))
        work = head
        while lst:
            new = ListNode(int(lst.pop()))
            work.next = new
            work = new
        return head
    
    return getLinked(getNum(l1) + getNum(l2))


    
"""
Linked List Cycle:
    Determine if there is a cycle. Told at most 10^4 nodes.
    This problem was meant to be solved with Floyd's algorithm in the 
    duplicate number solution below.  Was able to brute force O(n)
    time this since we were given n beforehand.
"""

def hasCycle(head: Optional[ListNode]) -> bool:
    if head == None:
        return False
    n = 1
    while head.next:
        head = head.next
        n += 1
        if n > 10**4:
            return True
    return False




"""
Find the Duplicate Number:
    Given an array of integers nums of length n+1 where each 
    integer is in the range [1, n] inclusive and exactly one number
    appears >= 2 times.  Find that number.
    
    View the array as a linked list where Node(k).next = Node(nums[k]) for k in 0,1,..,n
    The duplicated number with be the first number that repeats in the linked list.
    No need to actually make the linked list, since nums has the equivalent information.
    
    Floyd's algorithm can detect this number with O(1) extra memory and O(n) time.
    Naive solutions are memory*time = O(n^2). 
"""

def findDuplicate(nums: List[int]) -> int:
    
    fast, slow = 0, 0
    flag = True
    while flag:
        fast = nums[nums[fast]]
        slow = nums[slow]
        if fast == slow:
            flag=False
    slow = 0
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow   




"""
LRU Cache:

    Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    
    Implement the LRUCache class:
    
        LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
        int get(int key) Return the value of the key if the key exists, otherwise return -1.
        void put(int key, int value) Update the value of the key if the key exists. 
        Otherwise, add the key-value pair to the cache. If the number of keys exceeds the 
        capacity from this operation, evict the least recently used key.
    
    The functions get and put must each run in O(1) average time complexity.
"""

class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.dic = {} # will map key to node(key,val)
        
        # These will never be added to the dic, so their key/values do not matter
        self.left = Node(0,0)
        self.right = Node(0,0)
        
        self.left.next = self.right
        self.right.prev = self.left
        
    def remove(self,node):
        # The nodes to left/right of node
        pre, nex = node.prev, node.next
        # connect them together
        pre.next, nex.prev = nex, pre      
    
    def insert(self,node):
        # the nodes at the end
        pre, nex = self.right.prev, self.right
        
        # insert node between them
        node.prev, node.next = pre, nex
        pre.next = nex.prev = node
        
        self.dic[node.key] = node
        
    
    def get(self, key: int) -> int:
        if key in self.dic.keys():
            self.remove(self.dic[key])
            self.insert(self.dic[key])
            return self.dic[key].val
        return -1     
        
    def put(self, key: int, value: int) -> None:
        if key in self.dic.keys():
            self.remove(self.dic[key])
        
        # make new node and add it
        self.dic[key] = Node(key,value)
        self.insert(self.dic[key])
        
        # clear cache
        if len(self.dic) > self.cap:
            lru = self.left.next
            self.remove(lru)
            del self.dic[lru.key]

