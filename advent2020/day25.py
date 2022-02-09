#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 25
https://adventofcode.com/2020/day/25

This problem is an instance of Diffie–Hellman key exchange.

Alice and Bob publiclly agree on prime p and number g (with order p-1 in Z_p*)
(in this problem p = 20201227 and g = 7)

Alice and Bob secretely each pick private A and B and
publically transmit a = g^A mod p and b = g^B mod p
(in this problem a and b are the inputs)

Alice and Bob then compute: F = b^A = a^B mod p (since g^AB = g^BA)
(in this problem F is the answer)

A brief google search suggests there is no known clever way to recover F
from the public information of p, g, a, b.  The solution below
just tries each n until it finds g^n = a.
"""

test = [5764801, 17807724]
real = [10212254, 12577395]

p = 20201227
c_pub, d_pub = real[0], real[1]

# we know x_pub = 7**x_pri mod p
# we need to solve for either x_pri

# brute force
def pub_to_pri(pub0,pub1): # c_pub -> 1063911
    looking = True
    n = 0
    last = 1
    while looking and n < p:
        if last % p == pub0:
            looking = False
            return (0, n)
        elif last % p == pub1:
            looking = False
            return(1, n)
        else:
            last = last*7
            n = n+1

k_pri = 1063911

# the final answer is then d_pub**c_pri mod p
def trans(x,n):
    out = 1
    for i in range(n):
        out = (out*x) % p
    return out

card_secret = 1063911
print('Answer:')
print(trans(12577395, 1063911))