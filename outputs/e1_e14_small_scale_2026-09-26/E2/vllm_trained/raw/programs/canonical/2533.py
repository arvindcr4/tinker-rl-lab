
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
    def goodBinaryStrings(
        self, minLength: int, maxLength: int, oneGroup: int, zeroGroup: int
    ) -> int:
        mod = 10**9 + 7
        f = [1] + [0] * maxLength
        for i in range(1, len(f)):
            if i - oneGroup >= 0:
                f[i] += f[i - oneGroup]
            if i - zeroGroup >= 0:
                f[i] += f[i - zeroGroup]
            f[i] %= mod
        return sum(f[minLength:]) % mod

solution=Solution()
assert solution.goodBinaryStrings(3, 8, 2, 7) == 4
assert solution.goodBinaryStrings(1, 8, 3, 6) == 3
assert solution.goodBinaryStrings(5, 9, 6, 8) == 2
assert solution.goodBinaryStrings(1, 1, 1, 1) == 2
assert solution.goodBinaryStrings(2, 3, 3, 2) == 2
assert solution.goodBinaryStrings(2, 7, 2, 3) == 10
assert solution.goodBinaryStrings(5, 5, 3, 5) == 1
assert solution.goodBinaryStrings(3, 8, 4, 8) == 3
assert solution.goodBinaryStrings(1, 4, 3, 3) == 2
assert solution.goodBinaryStrings(1, 7, 4, 4) == 2
assert solution.goodBinaryStrings(1, 8, 4, 2) == 11
assert solution.goodBinaryStrings(5, 5, 5, 4) == 1
assert solution.goodBinaryStrings(5, 6, 5, 4) == 1
assert solution.goodBinaryStrings(1, 8, 6, 8) == 2
assert solution.goodBinaryStrings(1, 8, 4, 8) == 3
assert solution.goodBinaryStrings(2, 10, 8, 2) == 8
assert solution.goodBinaryStrings(3, 3, 2, 2) == 0
assert solution.goodBinaryStrings(4, 8, 1, 8) == 6
assert solution.goodBinaryStrings(5, 7, 6, 7) == 2
assert solution.goodBinaryStrings(4, 9, 7, 4) == 3
assert solution.goodBinaryStrings(4, 8, 5, 3) == 4
assert solution.goodBinaryStrings(5, 9, 2, 7) == 5
assert solution.goodBinaryStrings(2, 4, 4, 4) == 2
assert solution.goodBinaryStrings(3, 3, 3, 1) == 2
assert solution.goodBinaryStrings(3, 10, 2, 8) == 7
assert solution.goodBinaryStrings(5, 5, 1, 4) == 3
assert solution.goodBinaryStrings(5, 10, 6, 4) == 4
assert solution.goodBinaryStrings(4, 5, 1, 4) == 5
assert solution.goodBinaryStrings(3, 4, 4, 2) == 2
assert solution.goodBinaryStrings(2, 8, 2, 3) == 14
assert solution.goodBinaryStrings(5, 9, 3, 6) == 5
assert solution.goodBinaryStrings(2, 5, 5, 5) == 2
assert solution.goodBinaryStrings(4, 6, 5, 2) == 3
assert solution.goodBinaryStrings(3, 10, 2, 1) == 228
assert solution.goodBinaryStrings(4, 8, 4, 4) == 6
assert solution.goodBinaryStrings(1, 5, 3, 4) == 2
assert solution.goodBinaryStrings(1, 7, 7, 2) == 4
assert solution.goodBinaryStrings(2, 10, 6, 4) == 5
assert solution.goodBinaryStrings(5, 5, 2, 3) == 2
assert solution.goodBinaryStrings(5, 8, 6, 1) == 10
assert solution.goodBinaryStrings(1, 10, 5, 6) == 3
assert solution.goodBinaryStrings(4, 7, 4, 6) == 2
assert solution.goodBinaryStrings(3, 4, 1, 4) == 3
assert solution.goodBinaryStrings(4, 5, 2, 4) == 2
assert solution.goodBinaryStrings(2, 7, 3, 6) == 3
assert solution.goodBinaryStrings(2, 5, 5, 1) == 5
assert solution.goodBinaryStrings(4, 6, 4, 4) == 2
assert solution.goodBinaryStrings(2, 8, 3, 8) == 3
assert solution.goodBinaryStrings(4, 4, 3, 3) == 0
assert solution.goodBinaryStrings(3, 7, 3, 6) == 3
assert solution.goodBinaryStrings(3, 3, 2, 1) == 3
assert solution.goodBinaryStrings(3, 3, 1, 2) == 3
assert solution.goodBinaryStrings(3, 3, 1, 2) == 3
assert solution.goodBinaryStrings(5, 9, 3, 8) == 3
assert solution.goodBinaryStrings(4, 4, 2, 1) == 5
assert solution.goodBinaryStrings(3, 4, 4, 2) == 2
assert solution.goodBinaryStrings(2, 2, 1, 2) == 2
assert solution.goodBinaryStrings(3, 8, 1, 7) == 9
assert solution.goodBinaryStrings(4, 9, 7, 1) == 12
assert solution.goodBinaryStrings(1, 9, 5, 8) == 2
assert solution.goodBinaryStrings(5, 7, 7, 5) == 2
assert solution.goodBinaryStrings(2, 3, 2, 1) == 5
assert solution.goodBinaryStrings(5, 7, 7, 1) == 4
assert solution.goodBinaryStrings(5, 8, 6, 5) == 2
assert solution.goodBinaryStrings(4, 8, 7, 2) == 4
assert solution.goodBinaryStrings(3, 8, 4, 7) == 3
assert solution.goodBinaryStrings(3, 4, 3, 2) == 2
assert solution.goodBinaryStrings(5, 6, 1, 6) == 3
assert solution.goodBinaryStrings(1, 7, 4, 1) == 17
assert solution.goodBinaryStrings(5, 7, 2, 3) == 7
assert solution.goodBinaryStrings(1, 2, 1, 2) == 3
assert solution.goodBinaryStrings(3, 3, 1, 3) == 2
assert solution.goodBinaryStrings(4, 10, 2, 5) == 11
assert solution.goodBinaryStrings(4, 5, 2, 4) == 2
assert solution.goodBinaryStrings(3, 4, 2, 1) == 8
assert solution.goodBinaryStrings(1, 5, 4, 5) == 2
assert solution.goodBinaryStrings(4, 4, 2, 3) == 1
assert solution.goodBinaryStrings(5, 7, 6, 3) == 2
assert solution.goodBinaryStrings(4, 10, 7, 2) == 7
assert solution.goodBinaryStrings(5, 9, 5, 8) == 2
assert solution.goodBinaryStrings(5, 8, 8, 4) == 2
assert solution.goodBinaryStrings(4, 5, 4, 4) == 2
assert solution.goodBinaryStrings(5, 5, 3, 4) == 0
assert solution.goodBinaryStrings(5, 8, 1, 5) == 14
assert solution.goodBinaryStrings(1, 8, 1, 1) == 510
assert solution.goodBinaryStrings(3, 5, 2, 5) == 2
assert solution.goodBinaryStrings(4, 5, 4, 3) == 1
assert solution.goodBinaryStrings(4, 8, 7, 4) == 3
assert solution.goodBinaryStrings(2, 2, 1, 1) == 4
assert solution.goodBinaryStrings(4, 6, 1, 1) == 112
assert solution.goodBinaryStrings(2, 6, 6, 1) == 6
assert solution.goodBinaryStrings(5, 9, 1, 7) == 11
assert solution.goodBinaryStrings(3, 5, 4, 5) == 2
assert solution.goodBinaryStrings(2, 7, 7, 5) == 2
assert solution.goodBinaryStrings(4, 10, 4, 1) == 45
assert solution.goodBinaryStrings(4, 9, 1, 3) == 54
assert solution.goodBinaryStrings(1, 10, 5, 6) == 3
assert solution.goodBinaryStrings(5, 10, 3, 2) == 23
assert solution.goodBinaryStrings(4, 6, 2, 5) == 3
assert solution.goodBinaryStrings(5, 9, 7, 9) == 2