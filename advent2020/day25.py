#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 25
https://adventofcode.com/2020/day/25

This problem is an instance of Diffie–Hellman key exchange.

Alice and Bob publiclly agree on prime p and number g (with order p-1 in (Z/p)^*)
(in this problem p = 20201227 and g = 7)

Alice and Bob secretely each pick private A and B and
publically transmit a = g^A mod p and b = g^B mod p
(in this problem a and b are the inputs)

Alice and Bob then compute: F = b^A = a^B mod p (since g^AB = g^BA)
(in this problem F is the answer)

Below we implement the baby-step giant-step algorithm for solving
the discrete logorithm problem: given g^A mod p, recover A

With the private A we compute F = b^A mod p
"""
import numpy as np
  
def gcdE(a, b): # output solves gcd=a*x+b*y and hence a*x = gcd mod b 
    if a == 0 :  
        return b,0,1
             
    gcd,x1,y1 = gcdE(b%a, a) 
    
    x = y1 - (b//a) * x1 
    y = x1 
     
    return gcd,x,y    


def stepSTEP(n:int, g:int, a:int) -> int: # where g^output = a mod n  
    # algorithm works as long as a == g^(something)
    # if g does not generate (Z/n)^* then not all a's have this form

    # brute force search for x<n: g^x = a is O(n)
    # key observation is that solving g^x = a for x=im+j 
    # equivalent to solving g^j = a*(g^-m)^i for i,j < sqrt(n)
    
    # we compute each g^j and store as lookup[j] in hashtable in O(sqrt(n))
    # then brute force search for an i that works in O(sqrt(n))
    
    m = int(np.ceil(np.sqrt(n)))
    lookup = {0:1}
    while len(lookup) <= m:
        j = len(lookup)
        lookup[j] = ((lookup[j-1]*g) %n)
    
    # computes inverse of lookup[m] in (Z/n)^*
    s = gcdE(lookup[m], n)[1]
    
    for i in range(m):
        #print('big step', i*m)
        if (a*(s**i)) % n in lookup.values():
            for j in range(m):
                if lookup[j] == (a*(s**i)) % n:
                    return i*m + j
    
        
def power(x,b,n): # computes x^b mod n
    out = 1
    for i in range(b):
        out = (out*x) % n
    return out
    
test = [5764801, 17807724]
real = [10212254, 12577395]
p = 20201227

A = stepSTEP(p, 7, real[0])
b = real[1]

print('Answer:')    
print(power(b, A, p))    