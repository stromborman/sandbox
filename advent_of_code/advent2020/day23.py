#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 23
https://adventofcode.com/2020/day/23
"""

real = [int(i) for i in '389547612']
test = [int(i) for i in '389125467']


# Solution for part 1 is done naively
# the order of the cups is stored in a list
# crab's moves are done by manipulating the indices of said list


class Crab:
    def __init__(self, lst, m, pt=0):
        self.lst = [i % m for i in lst]
        self.pt = pt
        self.m = m
        
        
    def cycle(self):
        cur_cup = self.lst[self.pt]
        spotstomove = [(self.pt+i) % self.m for i in range(1,4)]
        numstomove = [self.lst[i] for i in spotstomove]
        
        new_lst = [i for i in self.lst if i not in numstomove]
        
        dest_cup = (cur_cup - 1) % self.m
        look = True
        while look:
            if dest_cup in new_lst:
                look = False
            else:
                dest_cup = (dest_cup - 1) % self.m
        
        loc = new_lst.index(dest_cup)+1
        
        self.lst = new_lst[:loc] + numstomove + new_lst[loc:]
        self.pt = (self.lst.index(cur_cup) + 1) % self.m
        
    def run(self, n):
        for i in range(n):
            self.cycle()
        loc1 = self.lst.index(1)
        return [self.lst[(loc1+j) % self.m] for j in range(1,9)]                


print('Answer to part1:')
print(Crab(real,9).run(100))

# In part 2, there are 1 million cups and we need to run 10 million rounds.
# The naive solution from part 1 'works' for part 2 but has a very long run time.
# This is because we are permuting lots of entires in a huge list each round.

# Luckily the permutations have a simple structure and only 3 cups have a new
# next cup after the permutation.  So we replace lst, our list of the cups' order,
# with a new list called next where next[lst[i]] = lst[i+1]
# For each round we only need to update next in 3 spots
# namely at the cur_cup, the end_cup, and the dest_cup

class BigCrab:
    def __init__(self, lst, m):
        lst = lst + [10]
        # below the number m in lst becomes the number 0
        # self.next defined so that next(lst[i]) = lst[i+1]
        self.next = [lst[0]] + [lst[lst.index(i)+1] for i in range(1,10)] + list(range(11,m)) + [0]
        self.cur_cup = lst[0]
        self.m = m
        
        
    def cycle(self):
        cur_cup = self.cur_cup
        
        # the 3 cups after the current cup
        three_cups = [self.next[cur_cup]]
        while len(three_cups) < 3:
            three_cups.append(self.next[three_cups[-1]])
        
        # the third of the the three_cups
        end_cup = three_cups[2]
        
        # the cup after which the three_cups are inserted
        dest_cup = (cur_cup - 1) % self.m
        look = True
        while look:
            if dest_cup not in three_cups:
                look = False
            else:
                dest_cup = (dest_cup - 1) % self.m
        
        # after the crab moves the cups, only three cups have a new next cup
        new_next_cur = self.next[end_cup] # cup that used to be after end_cup is now after cur_cup
        new_next_end = self.next[dest_cup] # cup that used to be after dest_cup is now after end_cup
        new_next_dest = self.next[cur_cup] # cup that used to be after cur_cup (ie three_cups[0]) is now after dest_cup
        
        self.next[cur_cup] = new_next_cur
        self.next[end_cup] = new_next_end
        self.next[dest_cup] = new_next_dest
                    
        self.cur_cup = self.next[self.cur_cup] # cup after cur_cup is the new cur_cup

        
    def run(self, n):
        for i in range(n):
            self.cycle()
        return self
            
    def ans(self):        
        num1 = self.next[1]
        num2 = self.next[num1]
        return num1*num2     

    # used for testing of self.next and self.cycle()    
    def recover(self):
        lst = [0]
        num = 0
        for i in range(self.m-1):
            lst.append(self.next[num])
            num = self.next[num]
        return lst

tbc= BigCrab(real, 1000000)

print('Answer to part2:')
print(tbc.run(10000000).ans())