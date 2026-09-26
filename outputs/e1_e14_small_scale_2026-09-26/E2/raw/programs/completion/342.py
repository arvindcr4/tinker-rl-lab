
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
    def isPowerOfFour(self, n: int) -> bool:
        # A number is a power of four if:
        # 1. It is positive
        # 2. It is a power of two (only one bit set)
        # 3. The single bit is in an odd position (1st, 3rd, 5th, etc.) when counting from right, starting at position 0
        #    This means the bit pattern is 1, 100, 10000, 1000000, etc.
        #    In binary: 1, 100, 10000, 1000000
        #    The mask for powers of two that are also powers of four is 0x55555555 (binary: 01010101010101010101010101010101)
        #    This mask has 1s in all odd positions (0-indexed from right)
        
        if n <= 0:
            return False
        
        # Check if n is a power of two: n & (n - 1) == 0
        # And check if the bit is in a position that corresponds to power of four
        # 0x55555555 is 01010101010101010101010101010101 in binary (32 bits)
        return (n & (n - 1)) == 0 and (n & 0x55555555) != 0

solution=Solution()
assert solution.isPowerOfFour(-132175863) == False
assert solution.isPowerOfFour(63926027) == False
assert solution.isPowerOfFour(-1411563909) == False
assert solution.isPowerOfFour(-112960369) == False
assert solution.isPowerOfFour(1398787527) == False
assert solution.isPowerOfFour(-2122873014) == False
assert solution.isPowerOfFour(-971922531) == False
assert solution.isPowerOfFour(499760520) == False
assert solution.isPowerOfFour(1923532046) == False
assert solution.isPowerOfFour(1353985302) == False
assert solution.isPowerOfFour(344341344) == False
assert solution.isPowerOfFour(988002889) == False
assert solution.isPowerOfFour(-1834052540) == False
assert solution.isPowerOfFour(-1341721426) == False
assert solution.isPowerOfFour(-1897807457) == False
assert solution.isPowerOfFour(1382286770) == False
assert solution.isPowerOfFour(-289903186) == False
assert solution.isPowerOfFour(-1909648519) == False
assert solution.isPowerOfFour(1211653495) == False
assert solution.isPowerOfFour(975806299) == False
assert solution.isPowerOfFour(-1687617090) == False
assert solution.isPowerOfFour(1455040039) == False
assert solution.isPowerOfFour(-1527326019) == False
assert solution.isPowerOfFour(-941091145) == False
assert solution.isPowerOfFour(1726122379) == False
assert solution.isPowerOfFour(-1454549046) == False
assert solution.isPowerOfFour(-1492737717) == False
assert solution.isPowerOfFour(681673362) == False
assert solution.isPowerOfFour(-498028798) == False
assert solution.isPowerOfFour(263604865) == False
assert solution.isPowerOfFour(-1857531484) == False
assert solution.isPowerOfFour(709452608) == False
assert solution.isPowerOfFour(-1428296390) == False
assert solution.isPowerOfFour(395219749) == False
assert solution.isPowerOfFour(-1776516426) == False
assert solution.isPowerOfFour(-514352355) == False
assert solution.isPowerOfFour(2039545274) == False
assert solution.isPowerOfFour(349232682) == False
assert solution.isPowerOfFour(167597989) == False
assert solution.isPowerOfFour(319856271) == False
assert solution.isPowerOfFour(864874249) == False
assert solution.isPowerOfFour(-578474853) == False
assert solution.isPowerOfFour(-991048998) == False
assert solution.isPowerOfFour(940794812) == False
assert solution.isPowerOfFour(-1685335034) == False
assert solution.isPowerOfFour(2143623148) == False
assert solution.isPowerOfFour(66135099) == False
assert solution.isPowerOfFour(2009648929) == False
assert solution.isPowerOfFour(-1128171337) == False
assert solution.isPowerOfFour(777732605) == False
assert solution.isPowerOfFour(1426286026) == False
assert solution.isPowerOfFour(1052197644) == False
assert solution.isPowerOfFour(1679825748) == False
assert solution.isPowerOfFour(-1866403904) == False
assert solution.isPowerOfFour(1222488442) == False
assert solution.isPowerOfFour(1971704535) == False
assert solution.isPowerOfFour(303366467) == False
assert solution.isPowerOfFour(-1316536413) == False
assert solution.isPowerOfFour(-1857993719) == False
assert solution.isPowerOfFour(834396323) == False
assert solution.isPowerOfFour(-29194969) == False
assert solution.isPowerOfFour(-1340825799) == False
assert solution.isPowerOfFour(-861107902) == False
assert solution.isPowerOfFour(-330957887) == False
assert solution.isPowerOfFour(354528954) == False
assert solution.isPowerOfFour(-843552779) == False
assert solution.isPowerOfFour(2050946487) == False
assert solution.isPowerOfFour(-902016635) == False
assert solution.isPowerOfFour(289265475) == False
assert solution.isPowerOfFour(392445895) == False
assert solution.isPowerOfFour(1495555314) == False
assert solution.isPowerOfFour(-981605610) == False
assert solution.isPowerOfFour(1101243042) == False
assert solution.isPowerOfFour(1323638794) == False
assert solution.isPowerOfFour(-508432525) == False
assert solution.isPowerOfFour(29953373) == False
assert solution.isPowerOfFour(1150779012) == False
assert solution.isPowerOfFour(-1737501345) == False
assert solution.isPowerOfFour(105307083) == False
assert solution.isPowerOfFour(-1501638540) == False
assert solution.isPowerOfFour(295765873) == False
assert solution.isPowerOfFour(-1250364443) == False
assert solution.isPowerOfFour(960304890) == False
assert solution.isPowerOfFour(468591715) == False
assert solution.isPowerOfFour(-1492755633) == False
assert solution.isPowerOfFour(1982634441) == False
assert solution.isPowerOfFour(2082588896) == False
assert solution.isPowerOfFour(1952603438) == False
assert solution.isPowerOfFour(-993187586) == False
assert solution.isPowerOfFour(497419628) == False
assert solution.isPowerOfFour(1136112891) == False
assert solution.isPowerOfFour(-1150533245) == False
assert solution.isPowerOfFour(-1840413329) == False
assert solution.isPowerOfFour(-1539356541) == False
assert solution.isPowerOfFour(-1045551619) == False
assert solution.isPowerOfFour(1822351967) == False
assert solution.isPowerOfFour(-1949501988) == False
assert solution.isPowerOfFour(1444861793) == False
assert solution.isPowerOfFour(1939352310) == False
assert solution.isPowerOfFour(465402098) == False