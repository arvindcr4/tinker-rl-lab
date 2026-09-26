
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

class Solution:
    def totalNQueens(self, n: int) -> int:
        def dfs(i):
            if i == n:
                nonlocal ans
                ans += 1
                return
            for j in range(n):
                a, b = i + j, i - j + n
                if cols[j] or dg[a] or udg[b]:
                    continue
                cols[j] = dg[a] = udg[b] = True
                dfs(i + 1)
                cols[j] = dg[a] = udg[b] = False

        cols = [False] * 10
        dg = [False] * 20
        udg = [False] * 20
        ans = 0
        dfs(0)
        return ans

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