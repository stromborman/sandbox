# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 16
adventofcode.com/2020/day/16
"""
import re

def read(filename):
    with open(filename) as file:
        sections = [sec.split('\n') for sec in file.read().split('\n\n')]
    return sections
        
        
test = read('input16t')
real = read('input16')

class Ticket:    
    def __init__(self, data, meaning=[]):
        self.data = list(map(int, data.split(',')))
        self.meaning = meaning
        
    def understand(self):
        return {self.meaning[i]:datum for i,datum in enumerate(self.data)}
            
    def nonsense_nums(self, rules):
        reg = re.compile(r'((\d+)-(\d+))') 
        lst = []
        for num in self.data:
            if not any([any([num in range(int(b),int(c)+1) for a,b,c in reg.findall(rule)]) for rule in rules]):
                lst.append(num)
        return lst
    
    def nonsense(self, rules):
        reg = re.compile(r'((\d+)-(\d+))')
        for num in self.data:
            if not any([any([num in range(int(b),int(c)+1) for a,b,c in reg.findall(rule)]) for rule in rules]):
                return True
        
        
def error_rate(lst):
    rules = lst[0]
    tickets = [Ticket(tick) for tick in lst[2][1:]]
    
    return sum([sum(ticket.nonsense_nums(rules)) for ticket in tickets])

print('Answer to part1:')
print(error_rate(real))

def error(lst):
    rules = lst[0]
    tickets = [Ticket(tick) for tick in lst[2][1:]]
    return [ticket for ticket in tickets if not ticket.nonsense(rules)]

real_ticks = error(real)

class Field:
    def __init__(self,entries,idn,meaning=''):
        self.entries = entries
        self.idn = idn
        self.meaning = meaning
    
    def check(self, rule) -> bool:
        return all([x in rule for x in self.entries])
    
class Rule:
    def __init__(self,raw):
        self.raw = raw
        self.about = raw.split(':')[0]
        self.bds = list(map(int,re.findall(r'(\d+)',raw)))
        self.set = set(range(self.bds[0], self.bds[1]+1)).union(set(range(self.bds[2],self.bds[3]+1)))
        
    
test1 = read('input16t1')
test_ticks = error(test1)


# fields = [f[1],f[2],...,f[n]] where f[i] is list of entries in ith spot
# rules = [R_1,R_2,...,R_n] where R_i[num] -> bool
# seek a permutation p in S_n so that
# R_p(i) returns true for all nums in f[i]

# will seek to find a permutation in stages
# in each stage we will do the following process
# for each f[i] let C_i be the set of R_j that are satisfied
# then for i with C_i the smallest pick an assignment R_j within C_i
# remove f[i] and R[j] and repeat this process

def decode(lst):
    ticks = error(lst)
    l = len(ticks[0].data)
    fields = [Field([tick.data[j] for tick in ticks],j) for j in range(l)]
    rules = [Rule(rule) for rule in lst[0]]
    
    # creates dict with possible rule assignments (value) for a given field (key)
    dct = {field: [rule for rule in rules if field.check(rule.set)] for field in fields} 
    
    # the permutation we are seeking
    matches = {}
    
    for n in range(len(fields)):
        small = min([len(x) for x in dct.values()])
        kpick = [key for key in dct if len(dct[key])==small][0]
        rpick = dct[kpick][0]
        matches[kpick] = rpick
        
        dct.pop(kpick)
        for key in dct.keys():
            dct[key][:] = [x for x in dct[key] if x!=rpick]
            
    # applies the matching so we now know the meaning of each field
    for field in matches:
        field.meaning = matches[field].about
        
    my_tick = Ticket(lst[1][1], [field.meaning for field in fields])
    
    # our ticket with meaning assigned to each number
    return my_tick.understand()

ans = 1
for key in decode(real):
    if key[0:9] == 'departure':
        ans = ans*decode(real)[key]


print('Answer to part2:')
print(ans)