
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
        # Before this block, we must have a good string of length i - j * oneGroup that ends with 0s (or the string starts here, i.e., i - j * oneGroup == 0).
        # So dp1[i] = sum over j >= 1 such that j * oneGroup <= i of:
        #   if i - j * oneGroup == 0: 1 (the string is all 1s)
        #   else: dp0[i - j * oneGroup]
        
        # Similarly, dp0[i] = sum over j >= 1 such that j * zeroGroup <= i of:
        #   if i - j * zeroGroup == 0: 1
        #   else: dp1[i - j * zeroGroup]
        
        # We can optimize this using prefix sums or by iterating efficiently.
        
        # Let's use a different approach:
        # dp[i] = total number of good binary strings of length i
        # dp[i] = dp1[i] + dp0[i]
        
        # dp1[i] = sum(dp0[i - k * oneGroup] for k in 1, 2, ... such that i - k * oneGroup >= 0)
        #         but when i - k * oneGroup == 0, we add 1 (empty prefix, which is valid as a starting point)
        # Similarly for dp0[i]
        
        # We can rewrite:
        # dp1[i] = sum(dp0[i - k * oneGroup] for k >= 1, i - k * oneGroup >= 0)
        #         where dp0[0] = 1 (base case: empty string ending with 0s doesn't make sense, but we can think of it as a sentinel)
        # Actually, let's define dp0[0] = 1 and dp1[0] = 1 as base cases for the empty prefix.
        # Then dp1[i] = sum(dp0[i - k * oneGroup] for k >= 1, i - k * oneGroup >= 0)
        # And dp0[i] = sum(dp1[i - k * zeroGroup] for k >= 1, i - k * zeroGroup >= 0)
        
        # But wait, the empty string isn't really a valid string. Let's think again.
        
        # Alternative formulation:
        # dp[i] = number of good binary strings of length i.
        # To form a good string of length i:
        # - It could end with a block of 1s of size oneGroup. Then the prefix of length i - oneGroup must be a good string that either is empty or ends with 0s.
        # - It could end with a block of 0s of size zeroGroup. Then the prefix of length i - zeroGroup must be a good string that either is empty or ends with 1s.
        
        # Let's define:
        # dp[i] = total good strings of length i
        # We'll compute dp[i] for i from 1 to maxLength.
        
        # Base case: dp[0] = 1 (empty string, as a base for building)
        
        # For i from 1 to maxLength:
        # dp[i] = 0
        # If i >= oneGroup:
        #   dp[i] += dp[i - oneGroup]  # Append a block of oneGroup 1s to any good string of length i - oneGroup
        # If i >= zeroGroup:
        #   dp[i] += dp[i - zeroGroup]  # Append a block of zeroGroup 0s to any good string of length i - zeroGroup
        
        # Wait, this isn't quite right because we need to ensure that we're not appending to a string that already ends with the same digit.
        
        # Let me reconsider. The key insight is:
        # A good string is a sequence of alternating blocks of 0s and 1s, where each block of 1s has size multiple of oneGroup, and each block of 0s has size multiple of zeroGroup.
        
        # Let dp[i] = number of good binary strings of length i.
        # Consider the last block:
        # - If the last block is 1s with size k * oneGroup (k >= 1), then the prefix of length i - k * oneGroup is a good string that either is empty or ends with 0s.
        # - If the last block is 0s with size k * zeroGroup (k >= 1), then the prefix of length i - k * zeroGroup is a good string that either is empty or ends with 1s.
        
        # Let's define:
        # dp[i] = total good strings of length i
        # dp1[i] = good strings of length i ending with 1s
        # dp0[i] = good strings of length i ending with 0s
        # dp[i] = dp1[i] + dp0[i]
        
        # dp1[i] = sum(dp0[i - k * oneGroup] for k >= 1, i - k * oneGroup >= 0)
        #         where dp0[0] = 1 (base case: empty prefix)
        # dp0[i] = sum(dp1[i - k * zeroGroup] for k >= 1, i - k * zeroGroup >= 0)
        #         where dp1[0] = 1 (base case: empty prefix)
        
        # We can optimize by noting:
        # dp1[i] = dp0[i - oneGroup] + dp0[i - 2*oneGroup] + ... 
        # dp1[i - oneGroup] = dp0[i - 2*oneGroup] + dp0[i - 3*oneGroup] + ...
        # So dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= 2*oneGroup
        # Actually, dp1[i] = dp0[i - oneGroup] + (dp1[i - oneGroup] if i >= 2*oneGroup else 0)
        # Hmm, let's be more careful.
        
        # dp1[i] = sum(dp0[i - k * oneGroup] for k >= 1, i - k * oneGroup >= 0)
        # dp1[i - oneGroup] = sum(dp0[i - oneGroup - k * oneGroup] for k >= 1, i - oneGroup - k * oneGroup >= 0)
        #                    = sum(dp0[i - (k+1) * oneGroup] for k >= 1, ...)
        #                    = sum(dp0[i - j * oneGroup] for j >= 2, i - j * oneGroup >= 0)
        # So dp1[i] = dp0[i - oneGroup] + dp1[i - oneGroup] if i >= oneGroup
        # But we need i >= oneGroup for dp1[i] to be non-zero.
        
        # Similarly, dp0[i] = dp1[i - zeroGroup] + dp0[i - zeroGroup] if i >= zeroGroup
        
        # Base cases:
        # dp1[0] = 0, dp0[0] = 0 (no string of length 0 ends with 1 or 0)
        # But for the recurrence, we need dp0[0] = 1 and dp1[0] = 1 as sentinels.
        
        # Let's set:
        # dp1[0] = 1, dp0[0] = 1 (sentinel values)
        # For i from 1 to maxLength:
        #   dp1[i] = 0
        #   if i >= oneGroup:
        #       dp1[i] = dp0[i - oneGroup]
        #       if i >= 2 * oneGroup:
        #           dp1
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