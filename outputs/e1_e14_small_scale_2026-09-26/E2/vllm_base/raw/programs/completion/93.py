
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
    def restoreIpAddresses(self, s: str) -> List[str]:
        """
        Given a string s containing only digits, return all possible valid IP addresses
        that can be formed by inserting dots into s.
        """
        result = []
        n = len(s)
        
        # An IP address has 4 parts, each part is 1-3 digits
        # So minimum length is 4, maximum is 12
        if n < 4 or n > 12:
            return result
        
        # Try all possible positions for the 3 dots
        # We need to place 3 dots to split the string into 4 parts
        # Let the lengths of the four parts be l1, l2, l3, l4 where each is 1, 2, or 3
        # And l1 + l2 + l3 + l4 = n
        
        def is_valid(segment: str) -> bool:
            """Check if a segment is a valid IP address part."""
            if len(segment) == 0:
                return False
            if len(segment) > 1 and segment[0] == '0':
                return False
            if len(segment) > 3:
                return False
            val = int(segment)
            if val < 0 or val > 255:
                return False
            return True
        
        def backtrack(start: int, parts: List[str]):
            """
            Recursively build IP address parts.
            start: current index in s
            parts: list of parts built so far
            """
            if len(parts) == 4:
                if start == n:
                    result.append(".".join(parts))
                return
            
            # Try taking 1, 2, or 3 digits for the next part
            for length in range(1, 4):
                if start + length > n:
                    break
                segment = s[start:start + length]
                if is_valid(segment):
                    parts.append(segment)
                    backtrack(start + length, parts)
                    parts.pop()
        
        backtrack(0, [])
        return result

solution=Solution()
assert solution.restoreIpAddresses('1335') == ['1.3.3.5']
assert solution.restoreIpAddresses('0287') == ['0.2.8.7']
assert solution.restoreIpAddresses('96857618') == ['96.85.76.18']
assert solution.restoreIpAddresses('56603852') == ['56.60.38.52']
assert solution.restoreIpAddresses('8844') == ['8.8.4.4']
assert solution.restoreIpAddresses('62905') == ['6.2.90.5', '6.29.0.5', '62.9.0.5']
assert solution.restoreIpAddresses('1986') == ['1.9.8.6']
assert solution.restoreIpAddresses('72460111') == ['7.24.60.111', '7.246.0.111', '72.4.60.111', '72.46.0.111']
assert solution.restoreIpAddresses('78491031') == ['78.49.10.31', '78.49.103.1']
assert solution.restoreIpAddresses('821042') == ['8.2.10.42', '8.2.104.2', '8.21.0.42', '8.210.4.2', '82.1.0.42', '82.10.4.2']
assert solution.restoreIpAddresses('16192') == ['1.6.1.92', '1.6.19.2', '1.61.9.2', '16.1.9.2']
assert solution.restoreIpAddresses('16140') == ['1.6.1.40', '1.6.14.0', '1.61.4.0', '16.1.4.0']
assert solution.restoreIpAddresses('247196756') == ['247.19.67.56', '247.196.7.56', '247.196.75.6']
assert solution.restoreIpAddresses('54851') == ['5.4.8.51', '5.4.85.1', '5.48.5.1', '54.8.5.1']
assert solution.restoreIpAddresses('2727') == ['2.7.2.7']
assert solution.restoreIpAddresses('286079') == ['2.8.60.79', '2.86.0.79', '28.6.0.79', '28.60.7.9']
assert solution.restoreIpAddresses('96148') == ['9.6.1.48', '9.6.14.8', '9.61.4.8', '96.1.4.8']
assert solution.restoreIpAddresses('5163') == ['5.1.6.3']
assert solution.restoreIpAddresses('41160') == ['4.1.1.60', '4.1.16.0', '4.11.6.0', '41.1.6.0']
assert solution.restoreIpAddresses('9431402') == ['9.43.140.2', '94.3.140.2', '94.31.40.2']
assert solution.restoreIpAddresses('56060') == ['5.6.0.60', '5.60.6.0', '56.0.6.0']
assert solution.restoreIpAddresses('6930') == ['6.9.3.0']
assert solution.restoreIpAddresses('942038254') == ['94.20.38.254', '94.203.8.254', '94.203.82.54']
assert solution.restoreIpAddresses('0914') == ['0.9.1.4']