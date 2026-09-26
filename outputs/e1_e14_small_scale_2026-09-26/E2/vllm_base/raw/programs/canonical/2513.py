
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
    def minimizeSet(
        self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int
    ) -> int:
        def f(x):
            cnt1 = x // divisor1 * (divisor1 - 1) + x % divisor1
            cnt2 = x // divisor2 * (divisor2 - 1) + x % divisor2
            cnt = x // divisor * (divisor - 1) + x % divisor
            return (
                cnt1 >= uniqueCnt1
                and cnt2 >= uniqueCnt2
                and cnt >= uniqueCnt1 + uniqueCnt2
            )

        divisor = lcm(divisor1, divisor2)
        return bisect_left(range(10**10), True, key=f)

solution=Solution()
assert solution.minimizeSet(10, 5, 3, 2) == 5
assert solution.minimizeSet(11, 9, 6, 1) == 7
assert solution.minimizeSet(10, 5, 10, 3) == 14
assert solution.minimizeSet(10, 6, 11, 9) == 20
assert solution.minimizeSet(8, 11, 11, 3) == 14
assert solution.minimizeSet(11, 2, 7, 6) == 13
assert solution.minimizeSet(2, 8, 5, 4) == 10
assert solution.minimizeSet(2, 2, 11, 8) == 37
assert solution.minimizeSet(6, 3, 8, 7) == 17
assert solution.minimizeSet(4, 9, 6, 11) == 17
assert solution.minimizeSet(6, 5, 8, 6) == 14
assert solution.minimizeSet(3, 7, 7, 5) == 12
assert solution.minimizeSet(5, 3, 11, 8) == 20
assert solution.minimizeSet(6, 11, 2, 8) == 10
assert solution.minimizeSet(4, 10, 5, 6) == 11
assert solution.minimizeSet(7, 4, 1, 9) == 11
assert solution.minimizeSet(6, 2, 8, 1) == 10
assert solution.minimizeSet(3, 6, 11, 10) == 25
assert solution.minimizeSet(9, 3, 1, 4) == 5
assert solution.minimizeSet(7, 8, 6, 9) == 15
assert solution.minimizeSet(2, 10, 10, 4) == 19
assert solution.minimizeSet(10, 8, 11, 7) == 18
assert solution.minimizeSet(4, 6, 10, 1) == 13
assert solution.minimizeSet(7, 10, 6, 10) == 16
assert solution.minimizeSet(2, 9, 11, 7) == 21
assert solution.minimizeSet(5, 10, 7, 9) == 17
assert solution.minimizeSet(10, 10, 4, 6) == 11
assert solution.minimizeSet(11, 5, 3, 7) == 10
assert solution.minimizeSet(9, 9, 7, 10) == 19
assert solution.minimizeSet(3, 8, 6, 9) == 15
assert solution.minimizeSet(6, 2, 8, 9) == 20
assert solution.minimizeSet(11, 10, 1, 5) == 6
assert solution.minimizeSet(2, 6, 2, 11) == 15
assert solution.minimizeSet(8, 4, 1, 3) == 4
assert solution.minimizeSet(7, 10, 10, 5) == 15
assert solution.minimizeSet(6, 3, 1, 6) == 8
assert solution.minimizeSet(8, 9, 9, 1) == 10
assert solution.minimizeSet(8, 9, 2, 1) == 3
assert solution.minimizeSet(9, 9, 10, 2) == 13
assert solution.minimizeSet(7, 9, 5, 5) == 10
assert solution.minimizeSet(6, 11, 10, 3) == 13
assert solution.minimizeSet(11, 11, 10, 3) == 14
assert solution.minimizeSet(5, 7, 1, 6) == 7
assert solution.minimizeSet(6, 10, 10, 2) == 12
assert solution.minimizeSet(4, 2, 5, 11) == 21
assert solution.minimizeSet(3, 11, 1, 7) == 8
assert solution.minimizeSet(6, 4, 4, 10) == 15
assert solution.minimizeSet(2, 9, 10, 5) == 19
assert solution.minimizeSet(6, 10, 10, 2) == 12
assert solution.minimizeSet(8, 6, 8, 8) == 16
assert solution.minimizeSet(10, 7, 10, 5) == 15
assert solution.minimizeSet(9, 2, 10, 2) == 12
assert solution.minimizeSet(2, 7, 4, 11) == 16
assert solution.minimizeSet(7, 6, 4, 1) == 5
assert solution.minimizeSet(9, 5, 11, 4) == 15
assert solution.minimizeSet(9, 3, 5, 6) == 12
assert solution.minimizeSet(3, 8, 3, 1) == 4
assert solution.minimizeSet(8, 6, 1, 10) == 11
assert solution.minimizeSet(5, 2, 1, 3) == 5
assert solution.minimizeSet(4, 10, 9, 6) == 15
assert solution.minimizeSet(8, 2, 3, 5) == 9
assert solution.minimizeSet(3, 11, 10, 3) == 14
assert solution.minimizeSet(10, 8, 11, 7) == 18
assert solution.minimizeSet(9, 10, 3, 3) == 6
assert solution.minimizeSet(2, 11, 7, 6) == 13
assert solution.minimizeSet(3, 9, 10, 7) == 19
assert solution.minimizeSet(7, 3, 1, 2) == 3
assert solution.minimizeSet(6, 2, 8, 8) == 19
assert solution.minimizeSet(5, 5, 3, 8) == 13
assert solution.minimizeSet(7, 10, 8, 9) == 17
assert solution.minimizeSet(4, 8, 7, 6) == 14
assert solution.minimizeSet(6, 7, 6, 9) == 15
assert solution.minimizeSet(4, 4, 8, 5) == 17
assert solution.minimizeSet(10, 2, 7, 6) == 14
assert solution.minimizeSet(6, 3, 5, 4) == 10
assert solution.minimizeSet(4, 5, 7, 6) == 13
assert solution.minimizeSet(10, 11, 4, 4) == 8
assert solution.minimizeSet(11, 5, 4, 6) == 10
assert solution.minimizeSet(5, 6, 11, 5) == 16
assert solution.minimizeSet(11, 10, 6, 1) == 7
assert solution.minimizeSet(6, 6, 3, 9) == 14
assert solution.minimizeSet(5, 5, 9, 11) == 24
assert solution.minimizeSet(5, 9, 1, 8) == 9
assert solution.minimizeSet(3, 8, 7, 11) == 18
assert solution.minimizeSet(2, 10, 6, 7) == 14
assert solution.minimizeSet(9, 8, 7, 6) == 13
assert solution.minimizeSet(3, 10, 10, 9) == 19
assert solution.minimizeSet(10, 7, 1, 5) == 6
assert solution.minimizeSet(4, 7, 4, 10) == 14
assert solution.minimizeSet(5, 8, 9, 5) == 14
assert solution.minimizeSet(3, 10, 1, 11) == 12
assert solution.minimizeSet(4, 5, 8, 9) == 17
assert solution.minimizeSet(11, 9, 7, 5) == 12
assert solution.minimizeSet(5, 4, 9, 3) == 12
assert solution.minimizeSet(10, 9, 11, 2) == 13
assert solution.minimizeSet(8, 11, 8, 6) == 14
assert solution.minimizeSet(3, 11, 10, 1) == 14
assert solution.minimizeSet(11, 3, 10, 10) == 20
assert solution.minimizeSet(3, 3, 10, 4) == 20
assert solution.minimizeSet(5, 4, 4, 10) == 14