#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 22
https://adventofcode.com/2020/day/22
"""

from collections import deque

def read(filename):
    with open(filename) as file:
        decks = file.read().split('\n\n')
        decks = [list(map(int, deck.split('\n')[1:])) for deck in decks]
        return decks
    
real = read('input22')
test = read('input22t')

class Game:
    def __init__(self, deck1, deck2):
        self.deck1 = deque(deck1)
        self.deck2 = deque(deck2)
        
    def around(self):
        card1 = self.deck1.popleft()
        card2 = self.deck2.popleft()
        
        if card1 > card2:
            self.deck1.extend([card1,card2])
        else:
            self.deck2.extend([card2,card1])
            
    def play(self):
        while len(self.deck1) > 0 and len(self.deck2) > 0:
            self.around()
        
        if len(self.deck1) == 0:
            winner = 2
            win_deck = list(self.deck2)
        
        if len(self.deck2) == 0:
            winner = 1
            win_deck = (self.deck1)
            
        l = len(win_deck)
        
        return winner, sum([c*(l-i) for i,c in enumerate(win_deck)])
        
        
print('Answer to part1:')
print(Game(real[0],real[1]).play())

class Rgame:
    def __init__(self, deck1, deck2, level, pre=[], win=None):
        self.deck1 = deque(deck1)
        self.deck2 = deque(deck2)
        self.pre = pre
        self.win = win
        self.level = level
        
    def around(self):
        if len(self.deck1) == 0:
            # print('player 2 WINS GAME by empty deck')
            self.win = 2
        elif len(self.deck2) == 0:
            # print('player 1 WINS GAME by empty deck')
            self.win = 1  
        elif [list(self.deck1), list(self.deck2)] in self.pre:
            # print('player 1 WINS GAME by no loops')
            self.win = 1
        else:
            self.pre.append([list(self.deck1), list(self.deck2)])
            card1 = self.deck1.popleft()
            card2 = self.deck2.popleft()
            # print('playing a round', card1, 'vs', card2)
            
            if card1 <= len(self.deck1) and card2 <= len(self.deck2):
                # print('intializing subgame at level', self.level+1)                
                deck1c = list(self.deck1)[0:card1]
                deck2c = list(self.deck2)[0:card2]
                
                roundwin = Rgame(deck1c, deck2c, self.level + 1, pre=[]).play()
                # print(roundwin, 'won round by winning subgame at level',self.level+1)
            
            else:
                if card1 > card2:
                    roundwin = 1
                else:
                    roundwin = 2
                # print(roundwin, 'won round via single combat')
                    
            if roundwin == 1:
                self.deck1.extend([card1,card2])
            elif roundwin == 2:
                self.deck2.extend([card2,card1])    
            
    def play(self):
        while self.win == None:
            self.around()
            
        if self.level >= 1:
            return self.win
        
        else:
            if self.win == 1:
                win_deck = list(self.deck1)
            else:
                win_deck = list(self.deck2)
        
            l = len(win_deck)
        
            return self.win, sum([c*(l-i) for i,c in enumerate(win_deck)])

# trg = Rgame(test[0], test[1], 0)

# print('winner is', trg.play())

print('Answer to part2:')
print(Rgame(real[0],real[1],0).play())
