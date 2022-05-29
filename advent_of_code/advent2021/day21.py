#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2021: Day 21
https://adventofcode.com/2021/day/21

Two player game played on Z/10 (1,2,..,10)
Starting locations for each player are randomly determined.
Starting scores are zero for each player.

On a player's turn they roll a dice three times, move the sum of
the rolls on the board, and then increase their score by the value
of their final location.  First player with at least 1000 points wins.
"""


"""
For part 1, we are given a deterministic d100 (rolls 1,2,3,...,100,1,2,...)
and P1 starts at 1, P2 starts at 3.  Need to determine the product of the
losing player's score with the number of dice rolls.
"""
from collections import deque
import numpy as np

def deterDie(turn, loc):
    board = [1,2,3,4,5,6,7,8,9,10]
    return board[(loc - 1 + 9*turn - 3) % 10]
    

class DeterministicGame:
    def __init__(self, P1loc, P2loc, P1score=0, P2score=0, turn=0):
        self.P1loc = P1loc
        self.P2loc = P2loc
        self.P1score = P1score
        self.P2score = P2score
        self.turn = turn
        
    def play(self):
        while max(self.P1score, self.P2score) < 1000:
            self.turn += 1
            if self.turn % 2 == 1:
                self.P1loc = deterDie(self.turn, self.P1loc)
                self.P1score += self.P1loc
                if self.P1score >= 1000:
                    loser_score = self.P2score
            else:
                self.P2loc = deterDie(self.turn, self.P2loc)
                self.P2score += self.P2loc
                if self.P2score >= 1000:
                    loser_score = self.P1score
        return 3*self.turn * loser_score
        

# testGame = DeterministicGame(4,8)
# print(testGame.play()) # 739785

realGame = DeterministicGame(1, 3)

print('Answer to part1:', realGame.play()) # 897798


"""
For part 2, we are given a d3, P1 starts at 1, P2 starts at 3, and first to >= 21 wins.
However for each roll multiple universes open up for each of the possible 3 rolls,
so each turn creates 27 universes though only 7 material different ones base on sum:
    3(1way), 4(3ways), 5(6ways), 6(7ways), 7(6ways), 8(3ways), 9(1ways)
Determine who wins in the most universes and how many times they win.
"""
die_outcomes = {3:1,
                4:3,
                5:6,
                6:7,
                7:6,
                8:3,
                9:1
                }

class QuantumGame:
    def __init__(self, locs, scores = [0,0], turn=0, number = 1):
        self.locs = locs
        self.scores = scores
        self.turn = turn
        self.number = number
        self.done = False if max(self.scores) < 21 else True
    
    def __repr__(self):
        return '[Locations: '+str(self.locs)+', '+\
                'Scores: '+str(self.scores)+', '+\
                'Instances: '+ str(self.number)+']\n'
                
    def addNumber(self, x):
        self.number = self.number + x
        return self
                
    def cleanDone(self):
        if self.done:
            if self.scores[0] >= 21:
                return np.array([self.number, 0])
            else:
                return np.array([0, self.number])
        else:
            return self
    
    def playTurn(self):
        def newLoc(loc,roll):
            board = [1,2,3,4,5,6,7,8,9,10]
            return board[(loc-1 + roll)%10]
        if self.done:
            return [self]
        else:
            keep_playing = []
            done_games = np.array([0,0])
            self.turn = (self.turn + 1) % 2
            for roll, ways in die_outcomes.items():
                if self.turn == 1:
                    newlocs = [newLoc(self.locs[0], roll), self.locs[1]]
                    newscores = [self.scores[0] + newlocs[0], self.scores[1]]
                else:
                    newlocs = [self.locs[0], newLoc(self.locs[1], roll)]
                    newscores = [self.scores[0], self.scores[1] + newlocs[1]]
                newgame = QuantumGame(newlocs, newscores, self.turn, self.number * ways)
                if newgame.done:
                    done_games += newgame.cleanDone()
                else:
                    keep_playing.append(newgame)
        return keep_playing, done_games
                    
        
def playout(locs):
    initial_game = QuantumGame(locs)
    wins = np.array([0,0])
    queue = deque([initial_game])
    while queue:
        # print(wins)
        game = queue.popleft()
        new_games, new_wins = game.playTurn()
        wins += new_wins
        for newgame in new_games:
            locs = newgame.locs
            scores = newgame.scores
            turn = newgame.turn
            flag = True
            for i, game in enumerate(queue):
                if flag:
                    if locs == game.locs and scores == game.scores and turn == game.turn:
                        queue[i] = game.addNumber(newgame.number)
                        flag = False
            if flag:
                queue.append(newgame)
    return wins
        
"""
Warning this runs for ~ 1 minute.  There are 100 boardstates * 400 scorestates * 2 turnstates,
but the markov chain is deterministic.  We could speed this up by recursively computing
the outcomes and caching them.  Eg if we know what happens from a game that start 
at (loc=[7,6], score=[7,6]), then we know what happens from (loc=[1,3],score=[0,0]) 
in the event of rolls 6 and 3 (and dont need to resimulate).
"""

part2 = playout([1,3])
       
print('Answer to part2:', max(part2)) # 48868319769358    