
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
    def maximumXOR(self, nums: List[int]) -> int:
        return reduce(or_, nums)

solution=Solution()
assert solution.maximumXOR([0, 88, 50, 67, 25, 11, 29, 91, 22, 51]) == 127
assert solution.maximumXOR([68, 63, 53]) == 127
assert solution.maximumXOR([36, 63, 80, 82, 79, 50, 60, 75, 74]) == 127
assert solution.maximumXOR([73, 69, 89, 41, 94, 10, 96]) == 127
assert solution.maximumXOR([69, 91, 77]) == 95
assert solution.maximumXOR([15, 21, 93, 16, 75, 43, 87, 40, 38]) == 127
assert solution.maximumXOR([4, 89, 42]) == 127
assert solution.maximumXOR([29, 78, 33]) == 127
assert solution.maximumXOR([59, 83, 97, 48, 51, 43, 53, 91, 93, 100]) == 127
assert solution.maximumXOR([16, 64, 31, 21, 10]) == 95
assert solution.maximumXOR([78, 66, 53, 75, 28, 72, 42]) == 127
assert solution.maximumXOR([98, 47, 49, 50, 62, 91]) == 127
assert solution.maximumXOR([85, 8, 58, 59, 53, 15, 83, 26, 46]) == 127
assert solution.maximumXOR([1, 61, 28, 57, 24, 97, 62, 49]) == 127
assert solution.maximumXOR([37, 96, 29, 92, 82, 22, 86, 15, 31, 68]) == 127
assert solution.maximumXOR([35, 46, 68, 21, 4, 99, 60, 81, 61, 67]) == 127
assert solution.maximumXOR([0, 47, 40, 98]) == 111
assert solution.maximumXOR([14, 47, 82, 17, 33, 54, 37, 85, 25]) == 127
assert solution.maximumXOR([88, 75, 35, 23, 85, 17, 20, 81]) == 127
assert solution.maximumXOR([95, 62, 8, 73, 71, 91, 51]) == 127
assert solution.maximumXOR([44, 70, 51, 91, 24, 42, 81, 88, 71]) == 127
assert solution.maximumXOR([85, 82]) == 87
assert solution.maximumXOR([5, 85, 84, 87, 34, 99, 17, 11, 27]) == 127
assert solution.maximumXOR([4, 37, 95, 26, 25]) == 127
assert solution.maximumXOR([1, 59, 67, 10, 93, 61]) == 127
assert solution.maximumXOR([34, 92, 35, 50]) == 127
assert solution.maximumXOR([30, 14]) == 30
assert solution.maximumXOR([80, 48, 31, 21, 52]) == 127
assert solution.maximumXOR([72, 24, 100, 6, 12, 2]) == 126
assert solution.maximumXOR([97, 78, 79, 14, 42, 88, 74, 2, 20, 76]) == 127
assert solution.maximumXOR([52, 6, 66, 62, 63, 0, 100]) == 127
assert solution.maximumXOR([29, 12, 82, 18, 26, 4, 84, 92, 13]) == 95
assert solution.maximumXOR([38, 54, 16]) == 54
assert solution.maximumXOR([78, 37, 53, 54, 73]) == 127
assert solution.maximumXOR([41, 76, 89, 46, 30, 36, 14, 68, 95]) == 127
assert solution.maximumXOR([29, 48, 50, 91, 64, 47]) == 127
assert solution.maximumXOR([62, 66, 88, 92, 68, 63, 33]) == 127
assert solution.maximumXOR([13, 41, 42, 77, 19]) == 127
assert solution.maximumXOR([35, 80, 58, 97, 81, 91, 44, 51, 8]) == 127
assert solution.maximumXOR([49, 98]) == 115
assert solution.maximumXOR([46, 86, 11, 79, 27, 99, 14, 7, 68, 88]) == 127
assert solution.maximumXOR([49, 23, 58, 0, 96, 54]) == 127
assert solution.maximumXOR([46, 94, 60, 1, 33, 58]) == 127
assert solution.maximumXOR([29, 71]) == 95
assert solution.maximumXOR([38, 36, 50]) == 54
assert solution.maximumXOR([97, 95, 18]) == 127
assert solution.maximumXOR([88, 14, 79, 3, 27, 51, 71, 77, 90]) == 127
assert solution.maximumXOR([57, 80, 43, 6, 49]) == 127
assert solution.maximumXOR([39, 25, 87, 20]) == 127
assert solution.maximumXOR([25, 45, 81, 62, 54]) == 127
assert solution.maximumXOR([24, 48, 5, 54, 32, 72, 12, 78, 6, 100]) == 127
assert solution.maximumXOR([3, 90, 99, 2, 40]) == 123
assert solution.maximumXOR([41, 74, 93, 82, 77, 44]) == 127
assert solution.maximumXOR([67, 56, 53, 69, 25, 66]) == 127
assert solution.maximumXOR([0, 57, 50, 61]) == 63
assert solution.maximumXOR([87, 51]) == 119
assert solution.maximumXOR([54, 98, 32, 40, 73]) == 127
assert solution.maximumXOR([7, 30, 9, 13, 1, 48]) == 63
assert solution.maximumXOR([73, 21, 0, 1, 70]) == 95
assert solution.maximumXOR([33, 59, 76]) == 127
assert solution.maximumXOR([64, 17, 63, 53, 72]) == 127
assert solution.maximumXOR([5, 19, 91, 39, 89]) == 127
assert solution.maximumXOR([40, 24, 15, 84, 79, 94, 29, 9, 51, 39]) == 127
assert solution.maximumXOR([19, 84, 86, 63, 46, 73, 7, 56]) == 127
assert solution.maximumXOR([46, 17, 48, 66, 89, 73, 71, 75, 55]) == 127
assert solution.maximumXOR([53, 86, 81, 44, 70, 4, 36]) == 127
assert solution.maximumXOR([96, 92, 32, 17, 27, 49, 79, 83]) == 127
assert solution.maximumXOR([78, 36, 43, 47, 60, 31, 19]) == 127
assert solution.maximumXOR([100, 24]) == 124
assert solution.maximumXOR([86, 50, 52, 41, 15, 69]) == 127
assert solution.maximumXOR([34, 85, 12, 40, 58, 24, 48]) == 127
assert solution.maximumXOR([83, 21, 62]) == 127
assert solution.maximumXOR([66, 86]) == 86
assert solution.maximumXOR([11, 52, 16, 99]) == 127
assert solution.maximumXOR([46, 60, 0, 7, 37, 21, 3]) == 63
assert solution.maximumXOR([66, 33, 99, 41, 18, 77, 38, 22, 53]) == 127
assert solution.maximumXOR([52, 19, 5]) == 55
assert solution.maximumXOR([44, 33, 27, 38, 26, 39]) == 63
assert solution.maximumXOR([91, 95, 81, 25, 21, 22, 51, 46, 45]) == 127
assert solution.maximumXOR([85, 63, 21, 98, 72, 53, 78, 96]) == 127
assert solution.maximumXOR([8, 43, 76, 50, 1, 17, 33, 37]) == 127
assert solution.maximumXOR([7, 27, 23, 41, 33, 0, 28, 20, 53, 8]) == 63
assert solution.maximumXOR([69, 95]) == 95
assert solution.maximumXOR([85, 15, 22, 99, 39]) == 127
assert solution.maximumXOR([13, 75, 54, 50, 9, 31, 85]) == 127
assert solution.maximumXOR([62, 49, 30, 43]) == 63
assert solution.maximumXOR([32, 52, 64, 81]) == 117
assert solution.maximumXOR([56, 1, 96, 3, 79, 99, 100]) == 127
assert solution.maximumXOR([43, 80, 23, 69, 39]) == 127
assert solution.maximumXOR([72, 92, 41]) == 125
assert solution.maximumXOR([9, 22, 12, 85, 38, 99, 39]) == 127
assert solution.maximumXOR([16, 46, 55, 84, 52, 87, 64, 40, 96, 18]) == 127
assert solution.maximumXOR([51, 13, 34, 70, 88, 65, 4]) == 127
assert solution.maximumXOR([81, 75]) == 91
assert solution.maximumXOR([37, 86, 83, 30]) == 127
assert solution.maximumXOR([47, 57]) == 63
assert solution.maximumXOR([41, 57, 61, 42, 29]) == 63
assert solution.maximumXOR([52, 7, 35, 90, 96]) == 127
assert solution.maximumXOR([96, 39, 45, 27, 81, 78, 56, 98]) == 127
assert solution.maximumXOR([19, 71, 6, 2, 44, 60, 3]) == 127