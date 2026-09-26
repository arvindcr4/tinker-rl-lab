
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
    def kInversePairs(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] = number of permutations of length i with exactly j inverse pairs
        # Base case: dp[0][0] = 1, dp[1][0] = 1
        # For a permutation of length i, when we insert the number i into a permutation of length i-1,
        # it can create 0 to i-1 new inverse pairs depending on its position.
        # dp[i][j] = sum(dp[i-1][j - t]) for t in range(0, min(j, i-1) + 1)
        
        # We use a 1D DP approach with prefix sums for efficiency
        # dp[j] will store the number of permutations of current length with j inverse pairs
        
        # Initialize for n=0 or n=1
        # dp[j] for permutations of length 1: only j=0 is possible with count 1
        dp = [0] * (k + 1)
        dp[0] = 1
        
        for i in range(2, n + 1):
            new_dp = [0] * (k + 1)
            # For each j, new_dp[j] = sum(dp[j - t]) for t in range(0, min(j, i-1) + 1)
            # This is equivalent to sum(dp[max(0, j - (i-1)) : j+1])
            # We can use a sliding window / prefix sum approach
            
            # Compute prefix sums of dp
            prefix = [0] * (k + 2)
            for j in range(k + 1):
                prefix[j + 1] = (prefix[j] + dp[j]) % MOD
            
            for j in range(k + 1):
                # t ranges from 0 to min(j, i-1)
                # So we need sum of dp[j - t] for t in [0, min(j, i-1)]
                # = sum of dp[m] for m in [j - min(j, i-1), j]
                # = sum of dp[m] for m in [max(0, j - (i-1)), j]
                left = max(0, j - (i - 1))
                right = j
                new_dp[j] = (prefix[right + 1] - prefix[left]) % MOD
            
            dp = new_dp
        
        return dp[k] % MOD

solution=Solution()
assert solution.kInversePairs(1, 2) == 0
assert solution.kInversePairs(1, 0) == 1
assert solution.kInversePairs(2, 2) == 0
assert solution.kInversePairs(2, 2) == 0
assert solution.kInversePairs(4, 3) == 6
assert solution.kInversePairs(2, 0) == 1
assert solution.kInversePairs(1, 3) == 0
assert solution.kInversePairs(8, 2) == 27
assert solution.kInversePairs(8, 2) == 27
assert solution.kInversePairs(8, 5) == 343
assert solution.kInversePairs(10, 5) == 1068
assert solution.kInversePairs(3, 2) == 2
assert solution.kInversePairs(4, 6) == 1
assert solution.kInversePairs(5, 6) == 20
assert solution.kInversePairs(9, 6) == 1230
assert solution.kInversePairs(5, 1) == 4
assert solution.kInversePairs(10, 5) == 1068
assert solution.kInversePairs(10, 4) == 440
assert solution.kInversePairs(5, 0) == 1
assert solution.kInversePairs(10, 6) == 2298
assert solution.kInversePairs(8, 6) == 602
assert solution.kInversePairs(4, 4) == 5
assert solution.kInversePairs(2, 2) == 0
assert solution.kInversePairs(2, 0) == 1
assert solution.kInversePairs(10, 3) == 155
assert solution.kInversePairs(10, 5) == 1068
assert solution.kInversePairs(9, 1) == 8
assert solution.kInversePairs(6, 2) == 14
assert solution.kInversePairs(8, 6) == 602
assert solution.kInversePairs(11, 5) == 1717
assert solution.kInversePairs(1, 4) == 0
assert solution.kInversePairs(2, 0) == 1
assert solution.kInversePairs(9, 5) == 628
assert solution.kInversePairs(5, 4) == 20
assert solution.kInversePairs(5, 1) == 4
assert solution.kInversePairs(2, 0) == 1
assert solution.kInversePairs(8, 6) == 602
assert solution.kInversePairs(9, 0) == 1
assert solution.kInversePairs(11, 4) == 649
assert solution.kInversePairs(11, 3) == 209
assert solution.kInversePairs(6, 2) == 14
assert solution.kInversePairs(11, 3) == 209
assert solution.kInversePairs(7, 5) == 169
assert solution.kInversePairs(1, 2) == 0
assert solution.kInversePairs(1, 6) == 0
assert solution.kInversePairs(7, 0) == 1
assert solution.kInversePairs(6, 4) == 49
assert solution.kInversePairs(10, 1) == 9
assert solution.kInversePairs(5, 2) == 9
assert solution.kInversePairs(3, 6) == 0
assert solution.kInversePairs(10, 1) == 9
assert solution.kInversePairs(1, 5) == 0
assert solution.kInversePairs(9, 6) == 1230
assert solution.kInversePairs(5, 5) == 22
assert solution.kInversePairs(10, 0) == 1
assert solution.kInversePairs(2, 0) == 1
assert solution.kInversePairs(7, 3) == 49
assert solution.kInversePairs(4, 1) == 3
assert solution.kInversePairs(1, 6) == 0
assert solution.kInversePairs(9, 2) == 35
assert solution.kInversePairs(3, 5) == 0
assert solution.kInversePairs(10, 3) == 155
assert solution.kInversePairs(11, 1) == 10
assert solution.kInversePairs(10, 5) == 1068
assert solution.kInversePairs(10, 0) == 1
assert solution.kInversePairs(2, 4) == 0
assert solution.kInversePairs(8, 2) == 27
assert solution.kInversePairs(1, 2) == 0
assert solution.kInversePairs(9, 5) == 628
assert solution.kInversePairs(1, 3) == 0
assert solution.kInversePairs(4, 5) == 3
assert solution.kInversePairs(10, 3) == 155
assert solution.kInversePairs(6, 3) == 29
assert solution.kInversePairs(10, 0) == 1
assert solution.kInversePairs(3, 0) == 1
assert solution.kInversePairs(10, 4) == 440
assert solution.kInversePairs(4, 3) == 6
assert solution.kInversePairs(5, 6) == 20
assert solution.kInversePairs(8, 1) == 7
assert solution.kInversePairs(2, 4) == 0
assert solution.kInversePairs(7, 6) == 259
assert solution.kInversePairs(1, 4) == 0
assert solution.kInversePairs(7, 2) == 20
assert solution.kInversePairs(4, 6) == 1
assert solution.kInversePairs(4, 5) == 3
assert solution.kInversePairs(11, 5) == 1717
assert solution.kInversePairs(4, 5) == 3
assert solution.kInversePairs(9, 3) == 111
assert solution.kInversePairs(2, 1) == 1
assert solution.kInversePairs(3, 5) == 0
assert solution.kInversePairs(8, 0) == 1
assert solution.kInversePairs(5, 4) == 20
assert solution.kInversePairs(10, 6) == 2298
assert solution.kInversePairs(8, 4) == 174
assert solution.kInversePairs(5, 5) == 22
assert solution.kInversePairs(2, 1) == 1
assert solution.kInversePairs(9, 0) == 1
assert solution.kInversePairs(9, 3) == 111
assert solution.kInversePairs(5, 3) == 15
assert solution.kInversePairs(9, 3) == 111