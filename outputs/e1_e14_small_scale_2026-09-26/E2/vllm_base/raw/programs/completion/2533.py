
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
    def goodBinaryStrings(self, minLength: int, maxLength: int, oneGroup: int, zeroGroup: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i] = number of good binary strings of length i
        # A good binary string ends with either a block of 1s or a block of 0s.
        # If it ends with a block of 1s, the last block has size k * oneGroup for some k >= 1.
        # If it ends with a block of 0s, the last block has size k * zeroGroup for some k >= 1.
        
        # We can think of it as:
        # dp[i] = (number of ways to form a good string of length i ending with 1s) + (number of ways to form a good string of length i ending with 0s)
        
        # Let dp1[i] = number of good binary strings of length i that end with a block of 1s
        # Let dp0[i] = number of good binary strings of length i that end with a block of 0s
        
        # For dp1[i]: The last block of 1s has size j * oneGroup where j >= 1 and j * oneGroup <= i.
        # Before this block, we must have a good string of length i - j * oneGroup that ends with 0s (or the string starts here).
        # So dp1[i] = sum over j >= 1 such that j*oneGroup <= i of:
        #   if i - j*oneGroup == 0: 1 (the string is all 1s)
        #   else: dp0[i - j*oneGroup]
        
        # Similarly, dp0[i] = sum over j >= 1 such that j*zeroGroup <= i of:
        #   if i - j*zeroGroup == 0: 1
        #   else: dp1[i - j*zeroGroup]
        
        # dp[i] = dp1[i] + dp0[i]
        
        # We can optimize using the fact that:
        # dp1[i] = dp0[i - oneGroup] + dp0[i - 2*oneGroup] + ... 
        # dp1[i - oneGroup] = dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ...
        # So dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup
        
        # Similarly, dp0[i] = dp1[i - zeroGroup] + dp0[i - zeroGroup] if i >= zeroGroup
        
        # Base cases:
        # dp1[0] = 0, dp0[0] = 0 (no string of length 0)
        # But we need to handle the case where the entire string is one block.
        
        # Let's redefine:
        # dp1[i] = number of good binary strings of length i ending with 1s
        # dp0[i] = number of good binary strings of length i ending with 0s
        
        # For i >= oneGroup:
        # dp1[i] = dp0[i - oneGroup] + (1 if i == oneGroup else 0) + dp1[i - oneGroup] if i >= 2*oneGroup... 
        # Actually, let's use the recurrence:
        # dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup, but we need to be careful.
        
        # Let me think again. 
        # dp1[i] = sum_{j>=1, j*oneGroup <= i} (1 if i - j*oneGroup == 0 else dp0[i - j*oneGroup])
        # = (1 if oneGroup == i else dp0[i - oneGroup]) + (1 if 2*oneGroup == i else dp0[i - 2*oneGroup]) + ...
        
        # dp1[i - oneGroup] = sum_{j>=1, j*oneGroup <= i - oneGroup} (1 if i - oneGroup - j*oneGroup == 0 else dp0[i - oneGroup - j*oneGroup])
        # = sum_{k>=2, k*oneGroup <= i} (1 if i - k*oneGroup == 0 else dp0[i - k*oneGroup])
        
        # So dp1[i] = (1 if i == oneGroup else dp0[i - oneGroup]) + dp1[i - oneGroup] if i >= oneGroup
        
        # Wait, let's verify:
        # dp1[i] = dp0[i - oneGroup] + dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ... + (1 if i == oneGroup else 0)
        # dp1[i - oneGroup] = dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ... + (1 if i - oneGroup == oneGroup else 0)
        #                  = dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ... + (1 if i == 2*oneGroup else 0)
        
        # So dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup, but we need to add 1 if i == oneGroup.
        # Actually, dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] for i >= oneGroup, where dp1[0] = 0, and we handle the base case separately.
        
        # Hmm, let's just use the direct recurrence with optimization:
        # dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup, with dp1[0] = 0
        # But this doesn't account for the case where the string is exactly one block of 1s.
        
        # Let me redefine more carefully:
        # Let dp1[i] = number of good binary strings of length i that end with a block of 1s.
        # For such a string, the last block of 1s has size k * oneGroup, k >= 1.
        # If k * oneGroup == i, then the string is all 1s, which is valid. Count = 1.
        # If k * oneGroup < i, then the prefix of length i - k * oneGroup must be a good string ending with 0s. Count = dp0[i - k * oneGroup].
        
        # So dp1[i] = (1 if i % oneGroup == 0 and i >= oneGroup else 0) + sum_{k>=2, k*oneGroup <= i} dp0[i - k*oneGroup]
        
        # Similarly, dp0[i] = (1 if i % zeroGroup == 0 and i >= zeroGroup else 0) + sum_{k>=2, k*zeroGroup <= i} dp1[i - k*zeroGroup]
        
        # Now, dp1[i] = (1 if i == oneGroup else 0) + dp0[i - oneGroup] + dp0[i - 2*oneGroup] + ...
        # dp1[i - oneGroup] = (1 if i - oneGroup == oneGroup else 0) + dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ...
        
        # So dp1[i] = (1 if i == oneGroup else 0) + dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup
        
        # Let's verify with i = oneGroup:
        # dp1[oneGroup] = 1 + dp0[0] + dp1[0] = 1 + 0 + 0 = 1. Correct.
        
        # For i = 2*oneGroup:
        # dp1[2*oneGroup] = 0 + dp0[oneGroup] + dp1[oneGroup]
        # dp0[oneGroup] = 0 if oneGroup < zeroGroup or oneGroup % zeroGroup != 0, etc.
        
        # This seems correct. Let's implement this.
        
        if minLength > maxLength:
            return 0
        
        dp1 = [0] * (maxLength + 1)
        dp0 = [0] * (maxLength + 1)
        
        for i in range(1, maxLength + 1):
            # Compute dp1[i]
            if i >= oneGroup:
                dp1[i] = dp0[i - oneGroup]
                if i >= 2 * oneGroup:
                    dp1[i] = (dp1[i] + dp1[i - oneGroup]) % MOD
                if i == oneGroup:
                    dp1[i] = (dp1[i] + 1) % MOD
            
            # Compute dp0[i]
            if i >= zeroGroup:
                dp0[i] = dp1[i - zeroGroup]
                if i >= 
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