# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 15
adventofcode.com/2020/day/15
"""
test = [0,3,6]
real = [20,0,1,11,6,3]

# key spoken number
# value is the two previous turns the number was spoken
# for a new number value is [turn,turn] 
def idct(lst):
    return {n:[i,i] for i,n in enumerate(lst)}

tdct = idct(test)
rdct = idct(real)

# returns the number spoken on turn n 
def last(lst, n):
    # setup stage of game
    dct = idct(lst)
    
    turn= len(lst) # current turn number
    num = lst[-1] # previous spoken number
    
    # while loop through the turns
    while turn < n:
        # when num is new
        if num not in dct.keys():
            dct[num] = [turn,turn]
        
        # speak the difference in turns for the last two times num was spoken  
        speak = dct[num][-1]-dct[num][-2]
        
        # update dict to account for speak being spoken
        if speak not in dct.keys():
            dct[speak]=[turn,turn]
        else:
            dct[speak] = [dct[speak][-1], turn]
        
        # update previous number spoken and move to next turn
        num = speak
        turn += 1

    return num


print('Answer to part1:')
print(last(real,2020))

print('Answer to part2:')
print(last(real,30000000))