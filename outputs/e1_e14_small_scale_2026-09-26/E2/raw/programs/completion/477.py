
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
    def totalHammingDistance(self, nums: List[int]) -> int:
        """
        Calculate the sum of Hamming distances between all pairs of integers in nums.
        
        Instead of computing each pair's Hamming distance individually (which would be O(n^2)),
        we can compute the contribution of each bit position separately.
        
        For each bit position (0 to 31), count how many numbers have that bit set (count_ones).
        The number of elements with that bit unset is (n - count_ones).
        The contribution to the total Hamming distance from this bit position is:
            count_ones * (n - count_ones)
        
        This is because each pair where one number has the bit set and the other doesn't
        contributes 1 to the Hamming distance for that bit position.
        """
        n = len(nums)
        total_distance = 0
        
        # Check each bit position from 0 to 31 (since nums[i] <= 10^9 < 2^30)
        for bit in range(32):
            count_ones = 0
            for num in nums:
                # Check if the bit-th bit is set
                if (num >> bit) & 1:
                    count_ones += 1
            # Each pair with different bits at this position contributes 1
            total_distance += count_ones * (n - count_ones)
        
        return total_distance

solution=Solution()
assert solution.totalHammingDistance([92, 41]) == 5
assert solution.totalHammingDistance([11, 53, 88]) == 14
assert solution.totalHammingDistance([21, 29, 67]) == 10
assert solution.totalHammingDistance([30, 21, 88, 58, 18, 97, 79]) == 80
assert solution.totalHammingDistance([21, 96, 92, 48, 5, 95, 72]) == 76
assert solution.totalHammingDistance([38, 71, 70, 24, 82, 96]) == 51
assert solution.totalHammingDistance([8, 18, 65, 37, 81, 67, 86, 61]) == 101
assert solution.totalHammingDistance([54, 87, 92, 65, 66, 58, 35]) == 80
assert solution.totalHammingDistance([89, 5, 52, 18]) == 23
assert solution.totalHammingDistance([25, 34, 3, 65, 44, 51, 58, 64, 32, 78]) == 156
assert solution.totalHammingDistance([7, 94, 17, 22]) == 19
assert solution.totalHammingDistance([71, 26]) == 5
assert solution.totalHammingDistance([24, 93, 1, 7, 15, 39]) == 49
assert solution.totalHammingDistance([97, 46, 88, 70, 19, 76, 91, 47, 48]) == 140
assert solution.totalHammingDistance([71, 66]) == 2
assert solution.totalHammingDistance([95, 58, 81, 12, 26, 10, 99]) == 76
assert solution.totalHammingDistance([90, 19, 47]) == 12
assert solution.totalHammingDistance([11, 70, 83, 38]) == 21
assert solution.totalHammingDistance([14, 78, 67, 76, 46, 38, 12, 88]) == 81
assert solution.totalHammingDistance([93, 44, 67, 74, 55, 62]) == 60
assert solution.totalHammingDistance([94, 42, 100, 76, 1]) == 38
assert solution.totalHammingDistance([19, 83]) == 1
assert solution.totalHammingDistance([72, 17, 91, 23, 42, 45, 19, 59]) == 96
assert solution.totalHammingDistance([27, 12, 76, 41, 82, 3, 84, 53]) == 106
assert solution.totalHammingDistance([94, 91, 74, 25, 49, 68, 99, 39, 18]) == 134
assert solution.totalHammingDistance([45, 40, 11, 6, 61, 81]) == 55
assert solution.totalHammingDistance([47, 60, 34, 94, 4, 77, 82, 98, 69]) == 134
assert solution.totalHammingDistance([87, 39]) == 3
assert solution.totalHammingDistance([86, 15, 13, 43, 85, 25]) == 52
assert solution.totalHammingDistance([3, 86, 39, 27, 78, 47, 80]) == 76
assert solution.totalHammingDistance([90, 65, 1, 14, 100, 84, 7, 85]) == 95
assert solution.totalHammingDistance([98, 32, 93, 12, 87, 45, 39]) == 80
assert solution.totalHammingDistance([92, 86]) == 2
assert solution.totalHammingDistance([52, 77, 26, 81, 2, 73, 99, 65, 45]) == 130
assert solution.totalHammingDistance([51, 33, 64, 18, 82, 91, 88, 69, 74, 19]) == 143
assert solution.totalHammingDistance([70, 42, 73, 97, 6]) == 36
assert solution.totalHammingDistance([59, 27, 9, 48, 31, 54, 87, 56, 55]) == 112
assert solution.totalHammingDistance([29, 34, 18, 59, 100, 9, 63, 64]) == 107
assert solution.totalHammingDistance([60, 65, 99, 97, 7, 36, 59, 15, 39, 27]) == 155
assert solution.totalHammingDistance([27, 17]) == 2
assert solution.totalHammingDistance([60, 91, 73, 24, 89, 59, 26]) == 58
assert solution.totalHammingDistance([90, 26, 16, 6, 78]) == 28
assert solution.totalHammingDistance([20, 66, 96]) == 10
assert solution.totalHammingDistance([46, 5, 8, 3, 99, 71, 68, 75, 87]) == 116
assert solution.totalHammingDistance([76, 91]) == 4
assert solution.totalHammingDistance([37, 69, 49, 10]) == 23
assert solution.totalHammingDistance([50, 59, 39, 38, 40]) == 28
assert solution.totalHammingDistance([30, 23, 53]) == 8
assert solution.totalHammingDistance([33, 32, 75, 3, 69, 82, 27, 94]) == 100
assert solution.totalHammingDistance([91, 53, 29, 38, 59, 4, 21, 76, 1]) == 126
assert solution.totalHammingDistance([74, 26, 89, 43, 72, 32]) == 47
assert solution.totalHammingDistance([36, 22, 84, 64, 76]) == 28
assert solution.totalHammingDistance([66, 15, 32, 5, 95, 96, 45]) == 78
assert solution.totalHammingDistance([20, 1]) == 3
assert solution.totalHammingDistance([30, 100, 52]) == 10
assert solution.totalHammingDistance([56, 31, 61]) == 8
assert solution.totalHammingDistance([49, 14, 94, 45, 34, 70, 74, 41]) == 104
assert solution.totalHammingDistance([11, 48, 97, 83, 79, 37, 87, 26, 75]) == 128
assert solution.totalHammingDistance([43, 86, 75, 27, 46, 50, 85, 48, 15, 20]) == 165
assert solution.totalHammingDistance([56, 57, 12, 51, 8, 17, 54, 97]) == 93
assert solution.totalHammingDistance([16, 36, 88, 75, 62, 63, 67, 40]) == 108
assert solution.totalHammingDistance([31, 63, 1, 74, 65, 14, 97, 8]) == 100
assert solution.totalHammingDistance([92, 68, 43, 60, 24, 93, 29, 47, 71]) == 124
assert solution.totalHammingDistance([95, 87, 94, 14, 34, 37, 19, 91]) == 96
assert solution.totalHammingDistance([9, 46, 69, 20, 99, 6, 58]) == 80
assert solution.totalHammingDistance([39, 95]) == 4
assert solution.totalHammingDistance([87, 11, 32, 35, 56, 94, 29, 60, 58, 52]) == 159
assert solution.totalHammingDistance([75, 3]) == 2
assert solution.totalHammingDistance([20, 52, 4, 42, 49, 86, 82, 57, 16]) == 114
assert solution.totalHammingDistance([28, 44]) == 2
assert solution.totalHammingDistance([53, 35, 59, 16, 33, 21, 17, 38, 32]) == 100
assert solution.totalHammingDistance([71, 99, 47, 68, 80, 24, 8, 2]) == 101
assert solution.totalHammingDistance([37, 39, 51]) == 6
assert solution.totalHammingDistance([61, 55]) == 2
assert solution.totalHammingDistance([81, 23]) == 3
assert solution.totalHammingDistance([87, 32, 29, 8, 42, 7]) == 57
assert solution.totalHammingDistance([96, 19, 43, 98, 61, 3, 2, 69, 46, 68]) == 159
assert solution.totalHammingDistance([14, 30, 36, 55, 81, 70]) == 54
assert solution.totalHammingDistance([38, 41, 66, 30, 14]) == 34
assert solution.totalHammingDistance([61, 41, 82, 46, 94, 13, 69, 89]) == 104
assert solution.totalHammingDistance([4, 90, 83, 22]) == 20
assert solution.totalHammingDistance([30, 83, 1, 15, 78, 21, 19, 14]) == 86
assert solution.totalHammingDistance([72, 29, 97, 34, 61, 64]) == 57
assert solution.totalHammingDistance([39, 52, 24, 86, 95, 20, 94]) == 68
assert solution.totalHammingDistance([78, 92, 61, 74, 5, 82, 80, 1, 26, 28]) == 152
assert solution.totalHammingDistance([59, 100, 51, 86, 64, 13]) == 62
assert solution.totalHammingDistance([73, 46, 96, 2, 78, 4, 61, 100, 84, 85]) == 156
assert solution.totalHammingDistance([42, 69, 84, 66, 40, 74, 35]) == 74
assert solution.totalHammingDistance([15, 1, 7, 32, 24, 81, 49, 34]) == 91
assert solution.totalHammingDistance([23, 100]) == 5
assert solution.totalHammingDistance([69, 77, 63, 12, 47, 27, 35, 40, 29, 28]) == 146
assert solution.totalHammingDistance([47, 35, 78, 75, 97, 10, 64, 5, 16, 56]) == 160
assert solution.totalHammingDistance([73, 72, 76]) == 4
assert solution.totalHammingDistance([53, 96]) == 4
assert solution.totalHammingDistance([10, 78, 82, 87, 73, 41, 75, 79, 88]) == 110
assert solution.totalHammingDistance([95, 49, 79, 90, 57, 4, 92, 53]) == 103
assert solution.totalHammingDistance([75, 22, 92, 78, 53, 30, 39, 96, 76]) == 130
assert solution.totalHammingDistance([58, 83]) == 4
assert solution.totalHammingDistance([70, 90, 72, 83, 66, 21, 74, 49, 98]) == 116
assert solution.totalHammingDistance([66, 8, 82, 72, 22, 75, 37, 93]) == 99