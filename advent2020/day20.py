#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Advent of Code 2020: Day 20
https://adventofcode.com/2020/day/20
"""

import queue
import itertools

def read(filename):
    with open(filename) as file:
        rawtiles = [tile.split('\n') for tile in file.read().split('\n\n')]
        tileDict = {int(tile[0][-5:-1]):tile[1:] for tile in rawtiles}
    return tileDict

def bd(lst): # 1->top, 2->bottom, 3->left, 4->right
    return [lst[0], lst[-1], ''.join([row[0] for row in lst]), ''.join([row[-1] for row in lst])]

def trans(lst): 
    return [''.join([row[j] for row in lst]) for j in range(len(lst))]

def flip(lst): #reflect across y-axis
    return [row[::-1] for row in lst]

def rot(lst): #rotate 90 anti-cw
    return trans(flip(lst))

def sym(lst,n): #i,r,r2,r3,f,fr,fr2,fr3
    out = lst
    if n % 4 != 0:
        for i in range(n % 4):
            out = rot(out)
    if n >= 4:
        out = flip(out)
    return out

class Tile:
    def __init__(self,id,data):
        self.id = id
        self.data = data
        self.bd = bd(self.data)
        self.loc = None
        self.nmatched = 0
        self.nmatches = 0
    
    # do self and tile fit together
    def match(self, tile, bo=True):
        if self == tile:
            if bo is True:
                False
            else:
                return []        
        else:
            if bo is True:
                return len([i for i in range(4) if (self.bd[i] in tile.bd or self.bd[i][::-1] in tile.bd)]) > 0
            else:
                return [i for i in range(4) if (self.bd[i] in tile.bd or self.bd[i][::-1] in tile.bd)]
    
    # get tiles that match with self from lst_tiles
    def matches(self, lst_tiles):
        return [tile for tile in lst_tiles if self.match(tile)]
        
    # finds proper orientation for tile to connect with self and adds tile to solution
    def solve(self, tile):
        if self.match(tile):
            edge_num = self.match(tile,bo=False)[0]           
            
            (i,j) = self.loc
                
            self.nmatched += 1
            tile.nmatched += 1
            
            if edge_num == 0:
                other_edge_num = 1
                tile.loc = (i-1,j)
            if edge_num == 1:
                other_edge_num = 0
                tile.loc = (i+1,j)
            if edge_num == 2:
                other_edge_num = 3
                tile.loc = (i,j-1)
            if edge_num == 3:
                other_edge_num = 2
                tile.loc = (i,j+1)
                
            found = False
            n = 0
            while found is False and n<8:
                if bd(sym(tile.data, n))[other_edge_num] == self.bd[edge_num]:
                    tile.data = sym(tile.data, n)
                    tile.bd = bd(tile.data)
                    found = True
                else:
                    n += 1
            # print(tile.id, 'added to solution on', self.id)
    
            
def loadTiles(filename):
    tileDict = read(filename)
    return {k:Tile(k,v) for k,v in tileDict.items()}
    
rTiles = loadTiles('input20')
tTiles = loadTiles('input20t')

ans = 1
for num in [tile.id for tile in rTiles.values() if len(tile.matches(rTiles.values())) == 2]:
    ans = ans*num

print('Answer to part1:')
print(ans)


def solver(tileDct,start=1):
    tiles = list(tileDct.values())
    for tile in tiles:
        tile.nmatches = len(tile.matches(tiles))
    
    intile = [tile for tile in tiles if tile.nmatches==2][start]
    intile.loc = (0,0)
    
    work = queue.PriorityQueue()
    work.put((2, intile.id))
    
    while work.empty() is False:
        now = work.get()
        tile = tileDct[now[1]]
        
        for new_tile in tile.matches(tiles):
            # print(new_tile.id, 'placed yet', new_tile.loc is not None)
            if new_tile.loc is None:
                tile.solve(new_tile)
                # print(new_tile.id, 'at', new_tile.loc)
                work.put((new_tile.nmatches - new_tile.nmatched, new_tile.id))

    return tileDct

rsolve = solver(rTiles,2)
tsolve = solver(tTiles,2)

def glue(tileDct):
    puz = {tile.loc: tile.data for tile in tileDct.values()}
    l = max([i for i,j in puz.keys()])+1
    
    glued = []
    for a in range(l):
        for i in range(1,9):
            row = ''
            for b in range(l):
                row = row+puz[(a,b)][i][1:9]
            glued.append(row)
            
    return glued
    

tfinal = glue(tsolve)
rfinal = glue(rsolve)


with open('input20s') as file:
    sea = file.readlines()[:-1]
sea_space = [(i,j) for i,j in itertools.product(range(3),range(20)) if sea[i][j]=='#']    

def monster(image):
    l = len(image)
    count = 0
    
    for a in range(0,l-3):
        for b in range(0,l-20):
            if all([image[a+i][b+j]=='#' for  i,j in sea_space]):
                count += 1

    return count

def find_monsters(image):
    looking = True
    n = 0
    while looking:
        how_many = monster(sym(image,n))
        if how_many > 0:
            looking = False
        else:
            n += 1
            
    return sum([row.count('#') for row in image]) - how_many*len(sea_space)

print('Answer to part2:')
print(find_monsters(rfinal))