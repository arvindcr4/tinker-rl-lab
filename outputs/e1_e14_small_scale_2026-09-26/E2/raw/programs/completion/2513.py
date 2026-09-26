
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
        
        # Let's think about what numbers are available:
        # - Numbers not divisible by divisor1: can go to arr1
        # - Numbers not divisible by divisor2: can go to arr2
        # - Numbers not divisible by both: can go to either arr1 or arr2
        # - Numbers divisible by both: can't go to either array
        
        # Let lcm = LCM(divisor1, divisor2)
        # Numbers divisible by both divisor1 and divisor2 are multiples of lcm.
        
        # For a given X, we need to check if it's possible to satisfy the conditions.
        
        # Count of numbers in [1, X] not divisible by divisor1: X - X // divisor1
        # Count of numbers in [1, X] not divisible by divisor2: X - X // divisor2
        # Count of numbers in [1, X] not divisible by either (i.e., not divisible by lcm): X - X // lcm
        
        # Let:
        # a = numbers only not divisible by divisor1 but divisible by divisor2 (can only go to arr1)
        # b = numbers only not divisible by divisor2 but divisible by divisor1 (can only go to arr2)
        # c = numbers not divisible by either (can go to either)
        # d = numbers divisible by both (can't go anywhere)
        
        # Actually, let's define:
        # - Only for arr1: numbers not divisible by divisor1 but divisible by divisor2
        # - Only for arr2: numbers not divisible by divisor2 but divisible by divisor1
        # - For both: numbers not divisible by either divisor1 or divisor2
        
        # Count of numbers in [1, X] divisible by divisor1: X // divisor1
        # Count of numbers in [1, X] divisible by divisor2: X // divisor2
        # Count of numbers in [1, X] divisible by both: X // lcm
        
        # Numbers not divisible by divisor1: X - X // divisor1
        # Numbers not divisible by divisor2: X - X // divisor2
        
        # Numbers that can go to arr1: not divisible by divisor1 = X - X // divisor1
        # Numbers that can go to arr2: not divisible by divisor2 = X - X // divisor2
        
        # But we need to ensure no overlap. Let's think more carefully.
        
        # Let's use inclusion-exclusion:
        # Numbers divisible by divisor1 OR divisor2: X // divisor1 + X // divisor2 - X // lcm
        # Numbers not divisible by either: X - (X // divisor1 + X // divisor2 - X // lcm)
        
        # For arr1, we need uniqueCnt1 numbers not divisible by divisor1.
        # For arr2, we need uniqueCnt2 numbers not divisible by divisor2.
        # These sets must be disjoint.
        
        # The key insight: 
        # - Numbers not divisible by divisor1 can be used for arr1.
        # - Numbers not divisible by divisor2 can be used for arr2.
        # - Numbers not divisible by either can be used for either.
        
        # Let's define:
        # total_not_div1 = X - X // divisor1  (can go to arr1)
        # total_not_div2 = X - X // divisor2  (can go to arr2)
        # total_not_either = X - X // divisor1 - X // divisor2 + X // lcm  (can go to either)
        
        # We need:
        # 1. total_not_div1 >= uniqueCnt1
        # 2. total_not_div2 >= uniqueCnt2
        # 3. The union of numbers used in arr1 and arr2 must be <= X, and they must be disjoint.
        
        # More precisely, let's think about it this way:
        # - Let A be the set of numbers in [1,X] not divisible by divisor1. |A| = X - X//divisor1
        # - Let B be the set of numbers in [1,X] not divisible by divisor2. |B| = X - X//divisor2
        # - We need to pick uniqueCnt1 from A and uniqueCnt2 from B such that the two sets are disjoint.
        
        # The numbers that can only go to arr1: A - B = numbers not divisible by divisor1 but divisible by divisor2
        # The numbers that can only go to arr2: B - A = numbers not divisible by divisor2 but divisible by divisor1
        # The numbers that can go to either: A ∩ B = numbers not divisible by either
        
        # count_only1 = |A - B| = |A| - |A ∩ B| = (X - X//div1) - (X - X//div1 - X//div2 + X//lcm) = X//div2 - X//lcm
        # count_only2 = |B - A| = |B| - |A ∩ B| = (X - X//div2) - (X - X//div1 - X//div2 + X//lcm) = X//div1 - X//lcm
        # count_both = |A ∩ B| = X - X//div1 - X//div2 + X//lcm
        
        # We need:
        # count_only1 >= 0, count_only2 >= 0 (always true)
        # We need to assign uniqueCnt1 to arr1 and uniqueCnt2 to arr2.
        # Let x1 = number from only1 assigned to arr1
        # Let x2 = number from only2 assigned to arr2
        # Let y1 = number from both assigned to arr1
        # Let y2 = number from both assigned to arr2
        # y1 + y2 <= count_both
        # x1 + y1 = uniqueCnt1
        # x2 + y2 = uniqueCnt2
        # x1 <= count_only1
        # x2 <= count_only2
        
        # So we need:
        # uniqueCnt1 - y1 <= count_only1 => y1 >= uniqueCnt1 - count_only1
        # uniqueCnt2 - y2 <= count_only2 => y2 >= uniqueCnt2 - count_only2
        # y1 + y2 <= count_both
        # y1 >= 0, y2 >= 0
        
        # So we need: max(0, uniqueCnt1 - count_only1) + max(0, uniqueCnt2 - count_only2) <= count_both
        
        # Also, we need count_only1 >= 0 and count_only2 >= 0 which is always true.
        # And we need uniqueCnt1 <= count_only1 + count_both = |A| = X - X//div1
        # And uniqueCnt2 <= count_only2 + count_both = |B| = X - X//div2
        
        # So the conditions are:
        # 1. X - X//div1 >= uniqueCnt1
        # 2. X - X//div2 >= uniqueCnt2
        # 3. max(0, uniqueCnt1 - count_only1) + max(0, uniqueCnt2 - count_only2) <= count_both
        
        # where count_only1 = X//div2 - X//lcm, count_only2 = X//div1 - X//lcm, count_both = X - X//div1 - X//div2 + X//lcm
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        lcm_val = lcm(divisor1, divisor2)
        
        def can_achieve(X):
            # Condition 1: enough numbers not divisible by divisor1 for arr1
            if X - X // divisor1 < uniqueCnt1:
                return False
            # Condition 2: enough numbers not divisible by divisor2 for arr2
            if X - X // divisor2 < uniqueCnt2:
                return False
            
            count_only1 = X // divisor2 - X // lcm_val
            count_only2 = X // divisor1 - X // lcm_val
            count_both = X - X // divisor1 - X // divisor2 + X // lcm_val
            
            # We need to cover the deficit from only1 and only2 using count_both
            deficit1 = max(0, uniqueCnt1 - count_only1)
            deficit2 = max(0, uniqueCnt2 - count_only2)
            
            if deficit1 + deficit2 > count_both:
                return False
            
            return True
        
        # Binary search for the minimum X

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