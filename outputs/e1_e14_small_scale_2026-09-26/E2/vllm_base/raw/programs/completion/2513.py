
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
import math

class Solution:
    def minimizeSet(self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int) -> int:
        # We need to find the minimum possible maximum integer X such that:
        # - arr1 has uniqueCnt1 distinct positive integers, none divisible by divisor1
        # - arr2 has uniqueCnt2 distinct positive integers, none divisible by divisor2
        # - No integer is in both arr1 and arr2
        # - All integers are in [1, X]
        
        # Let's define:
        # - A = numbers in [1, X] not divisible by divisor1 (candidates for arr1)
        # - B = numbers in [1, X] not divisible by divisor2 (candidates for arr2)
        # - C = numbers in [1, X] not divisible by divisor1 AND not divisible by divisor2 (can go to either)
        # - D1 = numbers in [1, X] divisible by divisor1 but not divisor2 (can only go to arr2)
        # - D2 = numbers in [1, X] divisible by divisor2 but not divisor1 (can only go to arr1)
        # - E = numbers in [1, X] divisible by both divisor1 and divisor2 (can't go to either)
        
        # Actually, let's think differently:
        # For a given X, we need to check if it's possible to assign numbers.
        
        # Count of numbers in [1, X] not divisible by divisor1: X - X // divisor1
        # Count of numbers in [1, X] not divisible by divisor2: X - X // divisor2
        
        # But we also need to ensure no overlap. Let's use inclusion-exclusion.
        
        # Numbers not divisible by divisor1: A = X - X // divisor1
        # Numbers not divisible by divisor2: B = X - X // divisor2
        
        # Numbers not divisible by either (i.e., not divisible by lcm(divisor1, divisor2)):
        lcm_val = math.lcm(divisor1, divisor2)
        not_div_by_either = X - X // lcm_val
        
        # Numbers divisible by divisor1 but not by divisor2: can only be in arr2
        # Numbers divisible by divisor2 but not by divisor1: can only be in arr1
        
        # Let's think about it this way:
        # - Numbers that can ONLY go to arr1: not divisible by divisor1, but divisible by divisor2
        #   Count = (X // divisor2) - (X // lcm_val)
        # - Numbers that can ONLY go to arr2: not divisible by divisor2, but divisible by divisor1
        #   Count = (X // divisor1) - (X // lcm_val)
        # - Numbers that can go to EITHER: not divisible by divisor1 AND not divisible by divisor2
        #   Count = X - X // divisor1 - X // divisor2 + X // lcm_val
        
        # For arr1: we need uniqueCnt1 numbers. Some can come from "only arr1" and some from "either".
        # For arr2: we need uniqueCnt2 numbers. Some can come from "only arr2" and some from "either".
        
        # Let:
        # only1 = X // divisor2 - X // lcm_val  (divisible by divisor2 but not divisor1, so can't be in arr1... wait)
        
        # Let me reconsider:
        # - arr1 cannot contain numbers divisible by divisor1
        # - arr2 cannot contain numbers divisible by divisor2
        
        # So:
        # - Numbers divisible by divisor1: can only go to arr2 (if not also divisible by divisor2... no, if divisible by divisor1, it can't be in arr1, but can be in arr2 only if NOT divisible by divisor2)
        # - Numbers divisible by divisor2: can only go to arr1 (if not also divisible by divisor1)
        # - Numbers divisible by both: can't go anywhere
        # - Numbers divisible by neither: can go to either
        
        # Let:
        # only_for_arr2 = numbers divisible by divisor1 but NOT by divisor2 = X//divisor1 - X//lcm_val
        # only_for_arr1 = numbers divisible by divisor2 but NOT by divisor1 = X//divisor2 - X//lcm_val
        # for_either = numbers not divisible by divisor1 AND not divisible by divisor2 = X - X//divisor1 - X//divisor2 + X//lcm_val
        
        # For arr1: we need uniqueCnt1 numbers. We can use at most only_for_arr1 numbers that are "forced" to arr1? No wait.
        # only_for_arr1 are numbers divisible by divisor2 but not divisor1. These CAN go to arr1 (since not divisible by divisor1). They CANNOT go to arr2 (since divisible by divisor2).
        # Similarly, only_for_arr2 are numbers divisible by divisor1 but not divisor2. These CAN go to arr2. They CANNOT go to arr1.
        
        # So:
        # - only_for_arr1 can ONLY be used in arr1
        # - only_for_arr2 can ONLY be used in arr2
        # - for_either can be used in either arr1 or arr2
        
        # For arr1 to have uniqueCnt1 elements:
        #   We must have: only_for_arr1 + (some from for_either) >= uniqueCnt1
        #   So: (some from for_either for arr1) >= uniqueCnt1 - only_for_arr1
        #   But we can't use negative, so max(0, uniqueCnt1 - only_for_arr1) from for_either
        
        # For arr2 to have uniqueCnt2 elements:
        #   Similarly: max(0, uniqueCnt2 - only_for_arr2) from for_either
        
        # Total needed from for_either: max(0, uniqueCnt1 - only_for_arr1) + max(0, uniqueCnt2 - only_for_arr2)
        # This must be <= for_either
        
        def can_achieve(X):
            only_for_arr1 = X // divisor2 - X // lcm_val
            only_for_arr2 = X // divisor1 - X // lcm_val
            for_either = X - X // divisor1 - X // divisor2 + X // lcm_val
            
            need1 = max(0, uniqueCnt1 - only_for_arr1)
            need2 = max(0, uniqueCnt2 - only_for_arr2)
            
            return need1 + need2 <= for_either
        
        # Binary search for the minimum X
        lo, hi = 1, 2 * 10**18  # Upper bound should be sufficient
        
        while lo < hi:
            mid = (lo + hi) // 2
            if can_achieve(mid):
                hi = mid
            else:
                lo = mid + 1
        
        return lo

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