
from typing import *
from bisect import *
from collections import *
from copy import *
from datetime import *
from heapq import *
from math import *
from re import *
from string import *
from random import *
from itertools import *
from functools import *
from operator import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import itertools
import functools
import operator


class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


from typing import *

class Solution:
    def totalNQueens(self, n: int) -> int:
        """
        Returns the number of distinct solutions to the n-queens puzzle.
        """
        # Use backtracking to count valid placements
        # We place queens row by row
        # Track columns and diagonals that are under attack
        
        count = 0
        # cols: set of column indices that have a queen
        # diag1: set of (row - col) values that have a queen (main diagonal)
        # diag2: set of (row + col) values that have a queen (anti-diagonal)
        cols = set()
        diag1 = set()  # row - col
        diag2 = set()  # row + col
        
        def backtrack(row: int):
            nonlocal count
            if row == n:
                count += 1
                return
            
            for col in range(n):
                d1 = row - col
                d2 = row + col
                
                # Check if this position is under attack
                if col in cols or d1 in diag1 or d2 in diag2:
                    continue
                
                # Place queen
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)
                
                # Recurse to next row
                backtrack(row + 1)
                
                # Backtrack: remove queen
                cols.remove(col)
                diag1.remove(d1)
                diag2.remove(d2)
        
        backtrack(0)
        return count

solution=Solution()
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(3) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(7) == 40
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(9) == 352
assert solution.totalNQueens(5) == 10
assert solution.totalNQueens(4) == 2
assert solution.totalNQueens(1) == 1
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(6) == 4
assert solution.totalNQueens(8) == 92
assert solution.totalNQueens(2) == 0
assert solution.totalNQueens(4) == 2