
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
    def maximumTastiness(self, price: List[int], k: int) -> int:
        price.sort()
        n = len(price)
        
        # Check if it's possible to pick k candies with minimum difference of 'diff'
        def can_achieve(diff: int) -> bool:
            count = 1
            last = price[0]
            for i in range(1, n):
                if price[i] - last >= diff:
                    count += 1
                    last = price[i]
                    if count >= k:
                        return True
            return False
        
        # Binary search on the answer
        low = 0
        high = price[-1] - price[0]
        result = 0
        
        while low <= high:
            mid = (low + high) // 2
            if can_achieve(mid):
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return result

solution=Solution()
assert solution.maximumTastiness([4, 13, 14, 16, 17], 4) == 1
assert solution.maximumTastiness([1, 1, 2, 3, 10, 16, 16, 18, 20], 4) == 4
assert solution.maximumTastiness([1, 6, 9, 15, 16, 17, 17, 17, 19, 19], 5) == 3
assert solution.maximumTastiness([6, 12], 2) == 6
assert solution.maximumTastiness([3, 8, 17, 17], 2) == 14
assert solution.maximumTastiness([1, 4, 11, 12], 2) == 11
assert solution.maximumTastiness([2, 9, 12, 13], 2) == 11
assert solution.maximumTastiness([9, 12, 12, 20, 20], 4) == 0
assert solution.maximumTastiness([2, 17], 2) == 15
assert solution.maximumTastiness([1, 1, 3, 5, 6, 9, 13, 13, 15, 17], 5) == 4
assert solution.maximumTastiness([1, 3, 4, 6, 10, 12], 3) == 5
assert solution.maximumTastiness([2, 3, 4, 6, 8, 10, 14], 6) == 2
assert solution.maximumTastiness([2, 9, 9, 19, 20], 5) == 0
assert solution.maximumTastiness([5, 9], 2) == 4
assert solution.maximumTastiness([3, 5, 6, 11, 12, 17, 18], 7) == 1
assert solution.maximumTastiness([2, 3, 8, 12, 15, 18, 18], 7) == 0
assert solution.maximumTastiness([9, 10, 11, 14, 16, 18], 4) == 2
assert solution.maximumTastiness([2, 8, 14, 15, 17, 18, 19, 20], 7) == 1
assert solution.maximumTastiness([11, 12, 13, 13, 17, 18, 18], 2) == 7
assert solution.maximumTastiness([4, 15, 20], 2) == 16
assert solution.maximumTastiness([2, 4, 7, 8, 13, 13, 15, 16, 17, 20], 10) == 0
assert solution.maximumTastiness([13, 20], 2) == 7
assert solution.maximumTastiness([19, 20], 2) == 1
assert solution.maximumTastiness([5, 6], 2) == 1
assert solution.maximumTastiness([1, 2, 2, 3, 6, 7, 10, 11, 12, 14], 2) == 13
assert solution.maximumTastiness([2, 4, 6, 8, 8, 10, 13, 16], 5) == 3
assert solution.maximumTastiness([2, 17, 19], 2) == 17
assert solution.maximumTastiness([11, 13, 14, 15, 16, 16, 19], 3) == 4
assert solution.maximumTastiness([12, 18, 19, 20], 4) == 1
assert solution.maximumTastiness([2, 10, 10, 14, 15], 4) == 1
assert solution.maximumTastiness([1, 2, 7, 7, 8, 9, 11, 12, 15], 4) == 4
assert solution.maximumTastiness([1, 10, 13, 13, 19], 2) == 18
assert solution.maximumTastiness([2, 5, 6, 13, 17], 5) == 1
assert solution.maximumTastiness([2, 6, 19, 19, 19], 2) == 17
assert solution.maximumTastiness([1, 6, 7, 11, 13, 13, 14, 16, 17, 20], 2) == 19
assert solution.maximumTastiness([8, 14], 2) == 6
assert solution.maximumTastiness([1, 4, 7, 7, 8, 14, 18], 7) == 0
assert solution.maximumTastiness([2, 2, 4, 6, 7, 10, 11, 12, 12, 16], 5) == 2
assert solution.maximumTastiness([3, 10, 14], 3) == 4
assert solution.maximumTastiness([6, 14, 19], 3) == 5
assert solution.maximumTastiness([1, 9, 15, 19], 3) == 8
assert solution.maximumTastiness([3, 6, 10, 14], 2) == 11
assert solution.maximumTastiness([1, 3, 4, 6, 6, 7, 10, 14, 15, 20], 4) == 6
assert solution.maximumTastiness([2, 2, 3, 4, 10, 16, 17], 2) == 15
assert solution.maximumTastiness([1, 3, 3, 3, 3, 4, 5, 7, 16], 3) == 6
assert solution.maximumTastiness([2, 3, 8, 10, 13, 17, 17], 4) == 4
assert solution.maximumTastiness([1, 13], 2) == 12
assert solution.maximumTastiness([6, 8, 10, 13, 15, 16, 20], 4) == 4
assert solution.maximumTastiness([3, 3, 13, 14, 17, 18], 3) == 5
assert solution.maximumTastiness([2, 3, 14, 17, 18], 2) == 16
assert solution.maximumTastiness([1, 3, 5, 8, 11], 4) == 3
assert solution.maximumTastiness([2, 4, 8, 12, 14, 15, 15, 16, 20, 20], 8) == 1
assert solution.maximumTastiness([2, 8, 10, 17], 4) == 2
assert solution.maximumTastiness([3, 6, 11, 14, 14, 16, 17, 20], 8) == 0
assert solution.maximumTastiness([1, 2, 8, 16, 17, 17, 17, 18], 2) == 17
assert solution.maximumTastiness([4, 9, 13], 3) == 4
assert solution.maximumTastiness([2, 3, 4, 7, 9, 12, 13, 16, 19], 8) == 1
assert solution.maximumTastiness([1, 2, 7, 8, 9, 10, 12, 17, 18, 19], 2) == 18
assert solution.maximumTastiness([3, 8, 9], 3) == 1
assert solution.maximumTastiness([6, 14], 2) == 8
assert solution.maximumTastiness([3, 9, 20], 2) == 17
assert solution.maximumTastiness([1, 2, 6, 9, 10, 11, 11, 12, 14, 17], 4) == 5
assert solution.maximumTastiness([4, 4, 5, 5, 5, 12, 15, 15, 15, 18], 3) == 6
assert solution.maximumTastiness([1, 4, 6, 6, 8, 9, 11, 12, 12, 18], 6) == 2
assert solution.maximumTastiness([2, 17], 2) == 15
assert solution.maximumTastiness([3, 6, 17], 2) == 14
assert solution.maximumTastiness([7, 10, 15], 2) == 8
assert solution.maximumTastiness([3, 13, 19], 3) == 6
assert solution.maximumTastiness([7, 11, 15, 16], 3) == 4
assert solution.maximumTastiness([2, 3, 3, 4, 6, 17, 18, 18], 4) == 2
assert solution.maximumTastiness([4, 4, 7, 8, 9, 11, 11, 12, 15], 8) == 0
assert solution.maximumTastiness([2, 3, 4, 6, 6, 15, 20, 20], 8) == 0
assert solution.maximumTastiness([2, 11, 13, 13, 16, 17, 20], 2) == 18
assert solution.maximumTastiness([2, 3, 3, 6, 10, 11, 13, 14, 16, 17], 9) == 1
assert solution.maximumTastiness([6, 8], 2) == 2
assert solution.maximumTastiness([4, 6, 18], 3) == 2
assert solution.maximumTastiness([5, 9, 14], 2) == 9
assert solution.maximumTastiness([11, 18], 2) == 7
assert solution.maximumTastiness([8, 8, 9, 10, 12, 14, 15, 15, 16, 20], 6) == 2
assert solution.maximumTastiness([9, 11, 12, 12, 14, 18, 20, 20], 4) == 2
assert solution.maximumTastiness([2, 4, 5, 8, 9, 10, 10, 11, 13, 18], 8) == 1
assert solution.maximumTastiness([7, 10, 13, 16, 19], 5) == 3
assert solution.maximumTastiness([5, 5, 5, 7, 12, 12, 12, 16, 17], 9) == 0
assert solution.maximumTastiness([5, 6, 12, 14, 15, 17, 20], 2) == 15
assert solution.maximumTastiness([3, 7, 8, 8, 8, 11, 13, 17], 3) == 6
assert solution.maximumTastiness([3, 5, 6, 8, 17, 17], 2) == 14
assert solution.maximumTastiness([3, 3, 5, 9, 10, 11, 11, 14, 16], 4) == 2
assert solution.maximumTastiness([3, 8, 15, 17, 19, 20], 4) == 5
assert solution.maximumTastiness([2, 6, 8], 3) == 2
assert solution.maximumTastiness([1, 4, 7, 8, 10, 11, 17, 17, 19], 6) == 2
assert solution.maximumTastiness([2, 3, 8, 9, 9, 14, 14, 17], 4) == 3
assert solution.maximumTastiness([2, 2, 3, 5, 5, 6, 7, 11, 15, 16], 7) == 1
assert solution.maximumTastiness([7, 13, 15, 17], 2) == 10
assert solution.maximumTastiness([2, 5, 6], 2) == 4
assert solution.maximumTastiness([2, 8, 10, 17], 3) == 7
assert solution.maximumTastiness([3, 5, 12, 13, 13, 13, 14, 14, 20, 20], 9) == 0
assert solution.maximumTastiness([7, 9, 10, 11, 17, 17, 18, 20], 7) == 1
assert solution.maximumTastiness([4, 10, 11, 17], 4) == 1
assert solution.maximumTastiness([3, 5, 8, 15, 20], 4) == 5
assert solution.maximumTastiness([3, 6, 6, 8, 14, 15, 19], 5) == 2