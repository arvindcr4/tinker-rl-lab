
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
        def is_valid(segment: str) -> bool:
            # Check if segment is valid
            if not segment:
                return False
            # Cannot have leading zeros unless the segment is exactly "0"
            if len(segment) > 1 and segment[0] == '0':
                return False
            # Must be between 0 and 255
            if int(segment) > 255:
                return False
            return True
        
        def backtrack(start: int, path: List[str]):
            # If we have 4 segments and we've used all characters, it's a valid IP
            if len(path) == 4:
                if start == len(s):
                    result.append(".".join(path))
                return
            
            # Try segments of length 1, 2, or 3
            for length in range(1, 4):
                if start + length > len(s):
                    break
                segment = s[start:start + length]
                if is_valid(segment):
                    path.append(segment)
                    backtrack(start + length, path)
                    path.pop()
        
        result = []
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