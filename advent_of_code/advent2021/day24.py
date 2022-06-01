#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 24
https://adventofcode.com/2021/day/24

In this problem we are provided a program that has 4 variables
and does arithmetic operations on given a set of inputs
which is a list 14 integers between 1 and 9, thought of
as a 14-digit integer.

An input passes if the 'z' variable, implimented as self.vars[3] below,
is equal to zero at the end of the program.

Part 1: Asks us to find the maximum passing number
Part 2: Asks us to fing the minimum passing number
"""
import re

def get_input(filename):
    with open(filename) as file:
        commands = file.read().split('\n')
        commands = [command.split(' ') for command in commands]
        for command in commands:
            if len(command) == 3:
                if re.match(r'[^wxyz]',command[2]):
                    command[2] = int(command[2])
    return commands
    
"""
A command in the list of commands is either ['inp','w'] (an input call)
or an operation ['opr', a, b] where a is a variable string and likewise for b (or just an integer)
"""

class ALU:        
    def __init__(self,instructions=[]):
        self.instructions = instructions
        self.vars = [0,0,0,0]
        
    def var_id(self, char): #'w'-> 0, ..'z'->3
        return ord(char) - ord('w')
    
    def var_val(self, var):
        if type(var) == int:
            return var
        else:
            return self.vars[self.var_id(var)]
    
    def operate(self,opr,a,b):
        if opr == 'add':
            return a + b
        if opr == 'mul':
            return a*b
        if opr == 'div':  
            return int(a/b)
        if opr == 'mod': 
            return a % b
        if opr == 'eql':
            return 1 if a == b else 0 
            
    def run(self,input_num:int,reset=True): # Output is the value of the 'z' variable, input passes if output is 0
        if reset:
            self.vars = [0,0,0,0]
        # we turn the input_num into a list of digits so we can feed the digits one by one
        inputs= list(str(input_num)) 
        for command in self.instructions:
            write_var = self.var_id(command[1])
            # only commands of length 2 are input calls
            if len(command) == 2:
            	# we pop the left most digit of the input_num and feed it to the input call
                self.vars[write_var] = int(inputs.pop(0))
            else:
                opr = command[0]
                a = self.var_val(command[1])
                b = self.var_val(command[2])
                self.vars[write_var] = self.operate(opr,a,b)
        return self.vars[3]
    
    # This method was for experimentation and understanding the different 'chunks' of the program
    def one_input(self,start,z,inp):
        self.vars = [inp,0,0,z]
        for command in self.instructions[start+1:]:
            write_var = self.var_id(command[1])
            if len(command) == 2:
                break
            else:
                opr = command[0]
                a = self.var_val(command[1])
                b = self.var_val(command[2])
                self.vars[write_var] = self.operate(opr,a,b)
        return self.vars[3]

test = get_input('input24t')
alut = ALU(test)
real = get_input('input24')
alu = ALU(real)
              

"""
Note that there is an input call every 18 commands.  Moreover, every
chunk of 18 commands is identical except for lines 4,5,15.

Line 4 has the two following types: 
    1(peek): div z 1
    2(pop): div z 26
Line 5 has the form add x b, (for some b) and x is zeroed out before this
Line 15 has the form add y c, (for some c) and y is zeroed out before this

The following parses each chunk and tells us what we have. 
"""
breaks = [i for i,command in enumerate(real) if len(command)==2]

def parse(chunk):
    out = [1,chunk[5][2],chunk[15][2]]
    if chunk[4][2] == 26:
        out[0] = 2
    return out

chunks = [parse(real[i*18:(i+1)*18]) for i in range(14)]
    
"""
These chunks operate as follows:
(1,b,c,i): z -> z*26 + i+c (unless i = z mod 26 + b, then z->z)
(2,b,c,i): z -> int(z/26)*26 + c (unless i = z mod 26 + b, then z->int(z/26))

All the 1's have b > 10 and i is always < 10, so unless never happens.
All the 2's have a negative b so unless is possible to meet.

Also note that all the b and c are abs < 26, so the entire story happens in base 26.
There are 7 type 1's and 7 type 2's, in order to get back to z = 0
we need the unless condition to be met in each type 2.

So the entire program can be simiplied to the following function, where
the output z is given in base 26.
"""

def actual(num):
    inputs= list(map(int,list(str(num))))
    stack = []
    for i in range(14):
        t,b,c = chunks[i]
        if t == 1:
            stack.append(inputs[i]+c)
        if t == 2:
            if inputs[i] != stack.pop() + b:
                stack.append(inputs[i]+c)
    return stack

"""
Since all we care about is how the digits pair up, the dictionary
computes the pairings, where dic[i] = (j,x) means
for input we must have input[i] = input[j] + x
"""

stack = []
dic = {}
for i in range(14):
    t,b,c = chunks[i]
    if t == 1:
        stack.append((i,c))
    if t == 2:
        j,cj = stack.pop()
        dic[i] = (j,cj+b)


"""
Since we will always have key > value[0] in the dic,
the size of the number is governed by the values and the keys
come for the ride.
"""

digits_links = [(value[0],key,value[1]) for key,value in dic.items()]

max_digits = [1]*14
for item in digits_links:
    j,i,x = item
    if x > 0:
        max_digits[j] = 9 - x
        max_digits[i] = 9
    if x <= 0:
        max_digits[j] = 9
        max_digits[i] = 9 + x

max_num = int(''.join(list(map(str,max_digits))))
actual(29989297949519)
alu.run(29989297949519)  

min_digits = [1]*14
for item in digits_links:
    j,i,x = item
    if x > 0:
        min_digits[j] = 1
        min_digits[i] = 1+x
    if x <= 0:
        min_digits[j] = 1-x
        min_digits[i] = 1

min_num = int(''.join(list(map(str,min_digits))))
actual(19518121316118)
alu.run(19518121316118)

print('Answer to part1:')
print(max_num)
print('Answer to part2:')
print(min_num)
