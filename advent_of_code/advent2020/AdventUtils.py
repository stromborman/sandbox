#shared utils file for common code
from typing import List

def read(filename: str) -> List[str]:
    """Given a FILE return a LIST of STRs"""
    with open(filename) as file:
        return [line.strip() for line in file.readlines()]

def parse_num(val: str) -> int:
    return int(val.strip('\n'))

def read_nums(filename) -> List[int]:
    """Given a FILE return a LIST of INTs"""
    lines = read(filename) #get list of file lines
    return list(map(parse_num, lines)) #parse each file line into an int

def read_nums_sorted(filename) -> List [int]:
    """Given a FILE return a LIST of sorted INTs"""
    lst = read_nums(filename)
    return sorted(lst+[0,max(lst)+3])
