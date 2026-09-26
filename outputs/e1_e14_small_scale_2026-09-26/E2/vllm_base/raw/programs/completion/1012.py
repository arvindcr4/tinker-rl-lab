
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
from math import factorial

class Solution:
    def numDupDigitsAtMostN(self, n: int) -> int:
        # We need to count numbers in [1, n] that have at least one repeated digit.
        # It's easier to count numbers with ALL DISTINCT digits, then subtract from n.
        
        def countDistinctDigits(k: int) -> int:
            """Count numbers with exactly k digits that have all distinct digits."""
            if k == 0:
                return 0
            if k == 1:
                return 10  # 0-9
            # First digit: 1-9 (9 choices)
            # Second digit: 0-9 except first digit (9 choices)
            # Third digit: 0-9 except first two (8 choices)
            # ...
            # k-th digit: 10 - (k-1) choices
            result = 9
            for i in range(1, k):
                result *= (10 - i)
            return result
        
        def countDistinctDigitsUpToK(k: int) -> int:
            """Count numbers with at most k digits that have all distinct digits."""
            total = 0
            for i in range(1, k + 1):
                total += countDistinctDigits(i)
            return total
        
        # Count numbers with distinct digits in [1, n]
        s = str(n)
        length = len(s)
        
        # Count all numbers with fewer digits than n that have distinct digits
        count = countDistinctDigitsUpToK(length - 1)
        
        # Count numbers with exactly 'length' digits that are <= n and have distinct digits
        # Use digit DP approach
        used = set()
        for i, ch in enumerate(s):
            digit = int(ch)
            # For position i, try digits from (start_digit) to digit-1
            # start_digit is 1 if i == 0, else 0
            start_digit = 1 if i == 0 else 0
            
            for d in range(start_digit, digit):
                if d in used:
                    continue
                # Count how many ways to fill the remaining positions with distinct digits
                remaining_positions = length - i - 1
                # We have used (i + 1) digits so far (including d)
                # Available digits: 10 - (i + 1)
                available = 10 - (i + 1)
                # Number of ways to choose and arrange remaining_positions from available
                if remaining_positions == 0:
                    ways = 1
                else:
                    if available < remaining_positions:
                        ways = 0
                    else:
                        ways = 1
                        for j in range(remaining_positions):
                            ways *= (available - j)
                count += ways
            
            # Now add the current digit and move to next position
            if digit in used:
                # Can't proceed further, break
                break
            used.add(digit)
            
            # If we've processed all digits, the number n itself has distinct digits
            if i == length - 1:
                count += 1
        
        # count is the number of integers in [1, n] with all distinct digits
        # The answer is n - count
        return n - count

solution=Solution()
assert solution.numDupDigitsAtMostN(148) == 27
assert solution.numDupDigitsAtMostN(498) == 120
assert solution.numDupDigitsAtMostN(543) == 129
assert solution.numDupDigitsAtMostN(123) == 23
assert solution.numDupDigitsAtMostN(767) == 191
assert solution.numDupDigitsAtMostN(102) == 11
assert solution.numDupDigitsAtMostN(654) == 159
assert solution.numDupDigitsAtMostN(557) == 139
assert solution.numDupDigitsAtMostN(359) == 85
assert solution.numDupDigitsAtMostN(847) == 214
assert solution.numDupDigitsAtMostN(38) == 3
assert solution.numDupDigitsAtMostN(227) == 49
assert solution.numDupDigitsAtMostN(752) == 187
assert solution.numDupDigitsAtMostN(132) == 24
assert solution.numDupDigitsAtMostN(45) == 4
assert solution.numDupDigitsAtMostN(600) == 150
assert solution.numDupDigitsAtMostN(589) == 147
assert solution.numDupDigitsAtMostN(204) == 39
assert solution.numDupDigitsAtMostN(394) == 92
assert solution.numDupDigitsAtMostN(933) == 240
assert solution.numDupDigitsAtMostN(701) == 178
assert solution.numDupDigitsAtMostN(512) == 124
assert solution.numDupDigitsAtMostN(364) == 86
assert solution.numDupDigitsAtMostN(441) == 103
assert solution.numDupDigitsAtMostN(423) == 98
assert solution.numDupDigitsAtMostN(735) == 184
assert solution.numDupDigitsAtMostN(537) == 129
assert solution.numDupDigitsAtMostN(546) == 131
assert solution.numDupDigitsAtMostN(405) == 95
assert solution.numDupDigitsAtMostN(987) == 249
assert solution.numDupDigitsAtMostN(929) == 239
assert solution.numDupDigitsAtMostN(520) == 125
assert solution.numDupDigitsAtMostN(869) == 219
assert solution.numDupDigitsAtMostN(117) == 19
assert solution.numDupDigitsAtMostN(488) == 119
assert solution.numDupDigitsAtMostN(780) == 201
assert solution.numDupDigitsAtMostN(487) == 118
assert solution.numDupDigitsAtMostN(292) == 64
assert solution.numDupDigitsAtMostN(853) == 215
assert solution.numDupDigitsAtMostN(669) == 171
assert solution.numDupDigitsAtMostN(818) == 209
assert solution.numDupDigitsAtMostN(829) == 211
assert solution.numDupDigitsAtMostN(715) == 180
assert solution.numDupDigitsAtMostN(607) == 151
assert solution.numDupDigitsAtMostN(805) == 206
assert solution.numDupDigitsAtMostN(438) == 101
assert solution.numDupDigitsAtMostN(466) == 115
assert solution.numDupDigitsAtMostN(443) == 105
assert solution.numDupDigitsAtMostN(626) == 155
assert solution.numDupDigitsAtMostN(560) == 141
assert solution.numDupDigitsAtMostN(994) == 256
assert solution.numDupDigitsAtMostN(22) == 2
assert solution.numDupDigitsAtMostN(70) == 6
assert solution.numDupDigitsAtMostN(32) == 2
assert solution.numDupDigitsAtMostN(177) == 33
assert solution.numDupDigitsAtMostN(694) == 175
assert solution.numDupDigitsAtMostN(856) == 216
assert solution.numDupDigitsAtMostN(554) == 136
assert solution.numDupDigitsAtMostN(132) == 24
assert solution.numDupDigitsAtMostN(779) == 201
assert solution.numDupDigitsAtMostN(65) == 5
assert solution.numDupDigitsAtMostN(863) == 217
assert solution.numDupDigitsAtMostN(154) == 28
assert solution.numDupDigitsAtMostN(632) == 155
assert solution.numDupDigitsAtMostN(564) == 141
assert solution.numDupDigitsAtMostN(621) == 153
assert solution.numDupDigitsAtMostN(629) == 155
assert solution.numDupDigitsAtMostN(758) == 189
assert solution.numDupDigitsAtMostN(646) == 159
assert solution.numDupDigitsAtMostN(436) == 101
assert solution.numDupDigitsAtMostN(697) == 176
assert solution.numDupDigitsAtMostN(765) == 189
assert solution.numDupDigitsAtMostN(486) == 118
assert solution.numDupDigitsAtMostN(124) == 23
assert solution.numDupDigitsAtMostN(293) == 64
assert solution.numDupDigitsAtMostN(196) == 36
assert solution.numDupDigitsAtMostN(859) == 217
assert solution.numDupDigitsAtMostN(795) == 203
assert solution.numDupDigitsAtMostN(158) == 29
assert solution.numDupDigitsAtMostN(424) == 99
assert solution.numDupDigitsAtMostN(508) == 123
assert solution.numDupDigitsAtMostN(53) == 4
assert solution.numDupDigitsAtMostN(99) == 9
assert solution.numDupDigitsAtMostN(483) == 117
assert solution.numDupDigitsAtMostN(713) == 180
assert solution.numDupDigitsAtMostN(427) == 99
assert solution.numDupDigitsAtMostN(577) == 145
assert solution.numDupDigitsAtMostN(45) == 4
assert solution.numDupDigitsAtMostN(237) == 53
assert solution.numDupDigitsAtMostN(561) == 141
assert solution.numDupDigitsAtMostN(250) == 55
assert solution.numDupDigitsAtMostN(230) == 51
assert solution.numDupDigitsAtMostN(918) == 236
assert solution.numDupDigitsAtMostN(316) == 69
assert solution.numDupDigitsAtMostN(956) == 244
assert solution.numDupDigitsAtMostN(702) == 178
assert solution.numDupDigitsAtMostN(929) == 239
assert solution.numDupDigitsAtMostN(783) == 201
assert solution.numDupDigitsAtMostN(39) == 3
assert solution.numDupDigitsAtMostN(346) == 83