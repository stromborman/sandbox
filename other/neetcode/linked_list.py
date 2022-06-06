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
Linked List Cycle:
    Determine if there is a cycle. Told at most 10^4 nodes.
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

   