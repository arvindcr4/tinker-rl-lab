
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
        
        # We can optimize this using prefix sums or by iterating properly.
        
        # Let's use a different approach:
        # dp[i] = total number of good binary strings of length i
        # dp[i] = dp1[i] + dp0[i]
        
        # dp1[i] = sum(dp0[i - k] for k in oneGroup, 2*oneGroup, 3*oneGroup, ... as long as i-k >= 0)
        #          plus 1 if i is a multiple of oneGroup (for the case where the entire string is 1s)
        # But wait, if i - k == 0, we add 1 (empty prefix, which is valid as a starting point)
        
        # Similarly for dp0[i]
        
        # Let's define:
        # dp1[i] = sum(dp0[i - j*oneGroup] for j >= 1 if i - j*oneGroup >= 0)
        #          where dp0[0] = 1 (base case: empty string ending with 0s doesn't make sense, but we can think of it as a sentinel)
        # Actually, let's set dp0[0] = 1 and dp1[0] = 1 as base cases for the "empty prefix" idea.
        # But that might double count. Let me reconsider.
        
        # Better approach:
        # dp[i] = number of good strings of length i
        # dp[i] = (sum of dp[i - j*oneGroup] for j >= 1, i - j*oneGroup >= 0, where we consider dp[0] = 1) 
        #         + (sum of dp[i - j*zeroGroup] for j >= 1, i - j*zeroGroup >= 0, where we consider dp[0] = 1)
        # But this counts strings ending in 1s and strings ending in 0s separately, and dp[0]=1 represents the empty string.
        
        # Wait, let's think again. If we define dp[i] as the total number of good strings of length i:
        # A good string of length i either:
        # 1. Ends with a block of 1s of size k*oneGroup. The prefix of length i - k*oneGroup is a good string (possibly empty).
        # 2. Ends with a block of 0s of size k*zeroGroup. The prefix of length i - k*zeroGroup is a good string (possibly empty).
        
        # But we need to be careful not to double count. A string can't end with both 1s and 0s.
        
        # So: dp[i] = sum_{j>=1, j*oneGroup<=i} dp[i - j*oneGroup] + sum_{j>=1, j*zeroGroup<=i} dp[i - j*zeroGroup]
        # where dp[0] = 1 (the empty string)
        
        # Let's verify with Example 1: minLength=2, maxLength=3, oneGroup=1, zeroGroup=2
        # dp[0] = 1
        # dp[1]: 
        #   From 1s: j=1, 1*1=1, dp[0]=1. So contribution = 1.
        #   From 0s: j=1, 1*2=2 > 1, no contribution.
        #   dp[1] = 1. String: "1"
        # dp[2]:
        #   From 1s: j=1, dp[1]=1; j=2, dp[0]=1. Contribution = 2.
        #   From 0s: j=1, dp[0]=1. Contribution = 1.
        #   dp[2] = 3. Strings: "11", "1", "00" -- wait, "1" has length 1, not 2.
        
        # Hmm, I think the issue is that dp[i] should only count strings of exactly length i.
        # When we say dp[i - j*oneGroup], if i - j*oneGroup = 0, we're saying the prefix is empty, so the string is just the block of 1s.
        # If i - j*oneGroup > 0, the prefix is a good string of that length.
        
        # Let me re-verify:
        # dp[0] = 1 (empty string)
        # dp[1] = dp[0] (from 1s, j=1) = 1. String: "1"
        # dp[2] = dp[1] + dp[0] (from 1s, j=1,2) + dp[0] (from 0s, j=1) = 1 + 1 + 1 = 3.
        #   From 1s: "11" (j=2, prefix empty), "1" + "1" but wait, "1" is a good string of length 1, appending "1" gives "11". 
        #   Actually, the recurrence is: for each good string of length i - j*oneGroup, append a block of j*oneGroup 1s.
        #   So for dp[2] from 1s:
        #     j=1: append "1" to good strings of length 1: "1" -> "11"
        #     j=2: append "11" to good strings of length 0: "" -> "11"
        #   Wait, that gives "11" twice! That's the problem.
        
        # I see the issue. The recurrence as stated double counts. Let me think differently.
        
        # Let's define:
        # dp1[i] = number of good strings of length i ending with 1
        # dp0[i] = number of good strings of length i ending with 0
        # dp[i] = dp1[i] + dp0[i]
        
        # dp1[i] = sum of dp0[i - j*oneGroup] for j >= 1, i - j*oneGroup >= 0
        #          where dp0[0] = 1 (base case: a string of all 1s of length j*oneGroup)
        # dp0[i] = sum of dp1[i - j*zeroGroup] for j >= 1, i - j*zeroGroup >= 0
        #          where dp1[0] = 1 (base case: a string of all 0s of length j*zeroGroup)
        
        # Let's verify:
        # dp1[0] = 1, dp0[0] = 1 (base cases)
        # dp[0] = 2? That doesn't seem right either.
        
        # Let me try yet another approach. Let's define dp[i] as the number of good strings of length i.
        # For a string of length i to be good:
        # - If it ends with 1, the last block of 1s has size k*oneGroup. The part
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