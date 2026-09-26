
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
        
        # Let's think about what numbers are available for each array.
        # For arr1: numbers in [1, X] that are NOT divisible by divisor1
        # For arr2: numbers in [1, X] that are NOT divisible by divisor2
        # But they must be disjoint.
        
        # Let's define:
        # A = set of numbers in [1, X] not divisible by divisor1
        # B = set of numbers in [1, X] not divisible by divisor2
        # We need to pick uniqueCnt1 from A and uniqueCnt2 from B such that the two sets are disjoint.
        
        # The key insight is to use binary search on X.
        # For a given X, we need to check if it's possible to satisfy the conditions.
        
        # Let's compute:
        # count1 = number of integers in [1, X] not divisible by divisor1
        # count2 = number of integers in [1, X] not divisible by divisor2
        # count_both = number of integers in [1, X] not divisible by divisor1 AND not divisible by divisor2
        # count_only1 = count1 - count_both (numbers not divisible by divisor1 but divisible by divisor2)
        # count_only2 = count2 - count_both (numbers not divisible by divisor2 but divisible by divisor1)
        
        # For arr1, we can use numbers from count_only1 and count_both.
        # For arr2, we can use numbers from count_only2 and count_both.
        # But count_both is shared, so we need to allocate count_both between arr1 and arr2.
        
        # Let x be the number of elements from count_both assigned to arr1.
        # Then (count_both - x) is assigned to arr2.
        # arr1 needs uniqueCnt1 elements: it can take at most count_only1 from only1, and x from both.
        # So: count_only1 + x >= uniqueCnt1 => x >= uniqueCnt1 - count_only1
        # Also x <= count_both and x >= 0.
        # Similarly for arr2: count_only2 + (count_both - x) >= uniqueCnt2 => x <= count_only2 + count_both - uniqueCnt2
        # Also count_both - x >= 0 => x <= count_both
        
        # So we need:
        # max(0, uniqueCnt1 - count_only1) <= x <= min(count_both, count_only2 + count_both - uniqueCnt2)
        # And this range must be valid, i.e., lower <= upper.
        
        # Alternatively, we can think:
        # The total numbers available for arr1 is count1, for arr2 is count2.
        # But they share count_both numbers.
        # The constraint is:
        # uniqueCnt1 <= count1 (obviously)
        # uniqueCnt2 <= count2 (obviously)
        # And uniqueCnt1 + uniqueCnt2 <= count1 + count2 - count_both + count_both = count1 + count2... no that's not right.
        
        # Let me re-derive:
        # Let S1 = numbers in [1,X] not div by d1
        # Let S2 = numbers in [1,X] not div by d2
        # |S1| = count1, |S2| = count2, |S1 ∩ S2| = count_both
        # |S1 ∪ S2| = count1 + count2 - count_both
        
        # We need to pick A ⊆ S1 with |A| = uniqueCnt1 and B ⊆ S2 with |B| = uniqueCnt2 such that A ∩ B = ∅.
        
        # The maximum number of elements we can pick for A from S1 \ S2 is |S1 \ S2| = count1 - count_both.
        # The maximum number of elements we can pick for B from S2 \ S1 is |S2 \ S1| = count2 - count_both.
        # The remaining elements for A must come from S1 ∩ S2, and similarly for B.
        
        # Let a = number of elements from S1 ∩ S2 assigned to A.
        # Let b = number of elements from S1 ∩ S2 assigned to B.
        # We need a + b <= count_both.
        # uniqueCnt1 = (elements from S1 \ S2) + a, so a = uniqueCnt1 - (elements from S1 \ S2)
        # But elements from S1 \ S2 <= count1 - count_both.
        # To maximize feasibility, we should use as many as possible from S1 \ S2 for A and S2 \ S1 for B.
        
        # So: a >= uniqueCnt1 - (count1 - count_both) = uniqueCnt1 - count1 + count_both
        # But a >= 0, so a >= max(0, uniqueCnt1 - count1 + count_both)
        # Similarly, b >= max(0, uniqueCnt2 - count2 + count_both)
        
        # And a + b <= count_both.
        
        # So the condition is:
        # max(0, uniqueCnt1 - count1 + count_both) + max(0, uniqueCnt2 - count2 + count_both) <= count_both
        
        # Let's verify with example 1: divisor1=2, divisor2=7, uniqueCnt1=1, uniqueCnt2=3, X=4
        # count1 = numbers in [1,4] not div by 2 = {1,3} = 2
        # count2 = numbers in [1,4] not div by 7 = {1,2,3,4} = 4
        # count_both = numbers in [1,4] not div by 2 and not div by 7 = {1,3} = 2
        # a >= max(0, 1 - 2 + 2) = max(0, 1) = 1
        # b >= max(0, 3 - 4 + 2) = max(0, 1) = 1
        # a + b = 2 <= count_both = 2. ✓
        
        # Example 2: divisor1=3, divisor2=5, uniqueCnt1=2, uniqueCnt2=1, X=3
        # count1 = numbers in [1,3] not div by 3 = {1,2} = 2
        # count2 = numbers in [1,3] not div by 5 = {1,2,3} = 3
        # count_both = numbers in [1,3] not div by 3 and not div by 5 = {1,2} = 2
        # a >= max(0, 2 - 2 + 2) = 2
        # b >= max(0, 1 - 3 + 2) = 0
        # a + b = 2 <= 2. ✓
        
        # Example 3: divisor1=2, divisor2=4, uniqueCnt1=8, uniqueCnt2=2, X=15
        # count1 = numbers in [1,15] not div by 2 = 8 (1,3,5,7,9,11,13,15)
        # count2 = numbers in [1,15] not div by 4 = 15 - 3 = 12 (all except 4,8,12)
        # count_both = numbers in [1,15] not div by 2 and not div by 4 = numbers not div by 2 = 8 (since if not div by 2, not div by 4)
        # a >= max(0, 8 - 8 + 8) = 8
        # b >= max(0, 2 - 12 + 8) = max(0, -2) = 0
        # a + b = 8 <= 8. ✓
        
        # So the check function for a given X is:
        # count1 = X - X // divisor1
        # count2 = X - X // divisor2
        # lcm = divisor1 * divisor2 // math.gcd(divisor1, divisor2)
        # count_both = X - X // divisor1 - X // divisor2 + X // lcm
        # a_min = max(0, uniqueCnt1 - count1 + count_both)
        # b_min = max(0, uniqueCnt2 - count2 + count_both)
        # return a_min + b
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