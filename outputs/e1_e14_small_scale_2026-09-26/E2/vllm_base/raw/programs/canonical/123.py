
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

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        f1, f2, f3, f4 = -prices[0], 0, -prices[0], 0
        for price in prices[1:]:
            f1 = max(f1, -price)
            f2 = max(f2, f1 + price)
            f3 = max(f3, f2 - price)
            f4 = max(f4, f3 + price)
        return f4

solution=Solution()
assert solution.maxProfit([68, 75, 16, 87, 23, 42, 92, 26]) == 140
assert solution.maxProfit([1, 69, 62, 96, 85]) == 102
assert solution.maxProfit([92, 79, 91, 62, 14, 66, 11, 78]) == 119
assert solution.maxProfit([9, 78, 25]) == 69
assert solution.maxProfit([98, 43, 61, 57, 45, 86]) == 59
assert solution.maxProfit([61, 27, 63, 72, 95, 68, 29]) == 68
assert solution.maxProfit([89, 77, 98, 26, 90, 97, 72, 75]) == 92
assert solution.maxProfit([86, 2, 98, 23, 15, 44, 7, 81, 56, 38]) == 170
assert solution.maxProfit([50, 83, 45, 34, 44]) == 43
assert solution.maxProfit([79, 75, 60, 64, 84, 88, 78, 69, 6]) == 28
assert solution.maxProfit([40, 97, 83, 4, 99, 28, 45, 26, 57, 2, 91]) == 184
assert solution.maxProfit([2, 76, 29, 90, 14, 12, 23, 30]) == 135
assert solution.maxProfit([74, 79, 43, 80, 37, 60, 69, 66]) == 69
assert solution.maxProfit([96, 18]) == 0
assert solution.maxProfit([2, 63, 11, 71, 10, 69, 73, 61, 43]) == 132
assert solution.maxProfit([84, 20]) == 0
assert solution.maxProfit([61, 31]) == 0
assert solution.maxProfit([44, 73, 38, 20, 8, 41, 48, 85, 33, 23]) == 106
assert solution.maxProfit([91, 49, 14, 81, 8, 61, 47, 85, 36, 9]) == 144
assert solution.maxProfit([26, 56, 92, 39, 74, 52, 25]) == 101
assert solution.maxProfit([92, 28, 100, 22]) == 72
assert solution.maxProfit([68, 43, 80, 52, 59, 95]) == 80
assert solution.maxProfit([3, 12]) == 9
assert solution.maxProfit([44, 99, 53]) == 55
assert solution.maxProfit([78, 32, 75]) == 43
assert solution.maxProfit([65, 26, 24, 92, 28, 42, 64, 93, 68, 39, 70, 61]) == 133
assert solution.maxProfit([9, 7]) == 0
assert solution.maxProfit([61, 14, 73, 75, 38, 10, 35, 87, 64]) == 138
assert solution.maxProfit([26, 31, 75, 50, 23]) == 49
assert solution.maxProfit([48, 20, 78, 98]) == 78
assert solution.maxProfit([61, 96, 71, 75, 8, 31]) == 58
assert solution.maxProfit([47, 15, 95, 2, 75, 11, 78, 68]) == 156
assert solution.maxProfit([32, 84, 93, 65]) == 61
assert solution.maxProfit([5, 57, 9, 42]) == 85
assert solution.maxProfit([29, 80, 74, 28, 8, 79, 37, 25, 64, 46, 21]) == 122
assert solution.maxProfit([46, 76, 96, 4, 47, 59, 11, 35, 61, 95, 86]) == 141
assert solution.maxProfit([94, 59, 78]) == 19
assert solution.maxProfit([11, 99, 1]) == 88
assert solution.maxProfit([79, 29, 16, 32, 69, 14, 27, 83, 18, 40, 4, 77]) == 142
assert solution.maxProfit([40, 31, 77, 75, 49, 89, 15, 35, 34]) == 86
assert solution.maxProfit([34, 80, 82, 1, 98, 10, 67, 58, 60, 24, 75, 74]) == 162
assert solution.maxProfit([28, 62, 70, 61, 14, 84, 13, 98, 12]) == 155
assert solution.maxProfit([87, 60, 26]) == 0
assert solution.maxProfit([48, 43, 15, 35, 83, 86, 92]) == 77
assert solution.maxProfit([32, 81, 56, 26, 19, 70, 37, 54, 25]) == 100
assert solution.maxProfit([53, 94, 99, 34, 15, 77, 4, 25, 49, 37, 89]) == 147
assert solution.maxProfit([96, 49, 20, 60]) == 40
assert solution.maxProfit([69, 51, 64, 83, 89]) == 38
assert solution.maxProfit([5, 100, 10, 7, 88, 85, 94, 57, 90, 62, 20, 81]) == 182
assert solution.maxProfit([27, 67, 39, 60, 51, 74, 56, 16, 42, 32, 85]) == 116
assert solution.maxProfit([89, 28, 27, 68, 20, 62, 74]) == 95
assert solution.maxProfit([48, 85, 27, 76, 55, 93, 33, 89, 51]) == 122
assert solution.maxProfit([97, 89, 88, 56, 69, 66, 13, 65, 55]) == 65
assert solution.maxProfit([35, 44, 40, 41, 3, 14, 76, 26]) == 82
assert solution.maxProfit([25, 57, 73, 44, 40, 61, 64, 7, 50, 22]) == 91
assert solution.maxProfit([11, 81, 66, 10, 12, 64, 93, 9, 86, 25, 50, 20]) == 160
assert solution.maxProfit([85, 79]) == 0
assert solution.maxProfit([8, 17, 75, 25, 40]) == 82
assert solution.maxProfit([23, 35]) == 12
assert solution.maxProfit([71, 41, 21, 11, 48, 1, 64, 35, 30, 76, 5, 36]) == 112
assert solution.maxProfit([57, 96, 30]) == 39
assert solution.maxProfit([77, 67, 62, 2, 32, 99, 41, 18, 76]) == 155
assert solution.maxProfit([70, 85, 32, 72, 57, 84, 30, 89]) == 111
assert solution.maxProfit([27, 60]) == 33
assert solution.maxProfit([96, 89, 63]) == 0
assert solution.maxProfit([42, 30, 85, 49]) == 55
assert solution.maxProfit([27, 98]) == 71
assert solution.maxProfit([79, 74, 92, 87, 85, 54, 2, 14]) == 30
assert solution.maxProfit([47, 27, 28, 22, 41, 100, 11, 90]) == 157
assert solution.maxProfit([70, 5, 65, 50, 10, 36]) == 86
assert solution.maxProfit([64, 26, 16, 55, 13, 60, 20, 24, 78, 31, 52, 70]) == 105
assert solution.maxProfit([34, 87, 81, 42, 63, 57, 10, 40, 66]) == 109
assert solution.maxProfit([20, 81, 85, 62, 75, 38, 33, 42, 100, 54, 16, 43]) == 132
assert solution.maxProfit([21, 5, 93, 63, 10, 99, 52, 20, 43, 11]) == 177
assert solution.maxProfit([51, 46, 62, 71, 5]) == 25
assert solution.maxProfit([33, 5, 16, 30, 7, 43, 68, 66, 35, 98, 65]) == 126
assert solution.maxProfit([100, 66, 34, 62, 31, 3, 26]) == 51
assert solution.maxProfit([10, 48, 23, 30, 88, 33, 71, 22]) == 116
assert solution.maxProfit([1, 90, 56, 40, 60, 24, 87, 82]) == 152
assert solution.maxProfit([57, 76, 71, 84, 25, 94, 83]) == 96
assert solution.maxProfit([30, 44, 29, 19, 37, 60, 25, 92, 50]) == 108
assert solution.maxProfit([10, 36, 62, 18, 7, 47, 70]) == 115
assert solution.maxProfit([54, 22, 37]) == 15
assert solution.maxProfit([89, 78, 9]) == 0
assert solution.maxProfit([89, 70, 83, 11, 5, 50, 74, 37, 18, 90, 66]) == 141
assert solution.maxProfit([46, 42, 31, 12, 93, 25, 80, 28, 26, 14, 61, 10]) == 136
assert solution.maxProfit([46, 82, 45]) == 36
assert solution.maxProfit([94, 43, 1, 47, 13]) == 46
assert solution.maxProfit([94, 11, 8, 31, 67, 64]) == 59
assert solution.maxProfit([64, 35, 21, 31, 60, 71, 98, 63, 69, 28, 16, 91]) == 152
assert solution.maxProfit([25, 11, 99, 26, 73]) == 135
assert solution.maxProfit([22, 11, 55, 44, 12, 87, 94, 78]) == 126
assert solution.maxProfit([65, 10, 58]) == 48
assert solution.maxProfit([41, 56, 24, 81, 59, 80, 4, 79, 6, 67]) == 136
assert solution.maxProfit([38, 31, 5, 36, 67, 60]) == 62
assert solution.maxProfit([65, 58, 87, 72, 69, 54, 82, 22, 37]) == 57
assert solution.maxProfit([59, 88, 13]) == 29
assert solution.maxProfit([84, 54, 18, 2, 3, 45, 88, 15]) == 86
assert solution.maxProfit([75, 43, 7, 12, 50, 81, 72, 84, 66, 82, 1]) == 93
assert solution.maxProfit([4, 77, 60, 73, 35, 3, 75, 7]) == 145