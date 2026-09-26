
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
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        n = len(security)
        if n <= time * 2:
            return []
        left, right = [0] * n, [0] * n
        for i in range(1, n):
            if security[i] <= security[i - 1]:
                left[i] = left[i - 1] + 1
        for i in range(n - 2, -1, -1):
            if security[i] <= security[i + 1]:
                right[i] = right[i + 1] + 1
        return [i for i in range(n) if time <= min(left[i], right[i])]

solution=Solution()
assert solution.goodDaysToRobBank([7, 80, 45, 89, 41, 14, 84, 6], 15) == []
assert solution.goodDaysToRobBank([28, 39, 32, 1, 17, 75, 68], 154) == []
assert solution.goodDaysToRobBank([56, 93, 33, 59, 91, 67, 35, 52, 19], 60) == []
assert solution.goodDaysToRobBank([43, 61, 39, 30, 63, 60, 51, 76, 82], 104) == []
assert solution.goodDaysToRobBank([92, 95, 38, 54], 175) == []
assert solution.goodDaysToRobBank([52, 74, 5, 79, 61], 61) == []
assert solution.goodDaysToRobBank([24, 52, 29, 3, 73, 80, 67, 77, 60], 72) == []
assert solution.goodDaysToRobBank([43, 62, 64, 11, 18, 71], 143) == []
assert solution.goodDaysToRobBank([100, 63, 67, 6, 74, 25, 53], 153) == []
assert solution.goodDaysToRobBank([95, 23, 48, 70, 82, 18, 13, 86, 100, 84], 128) == []
assert solution.goodDaysToRobBank([5, 4, 49, 13, 56, 15], 55) == []
assert solution.goodDaysToRobBank([36, 19, 23, 39], 154) == []
assert solution.goodDaysToRobBank([72, 5, 17], 194) == []
assert solution.goodDaysToRobBank([7, 29, 72, 5, 4], 26) == []
assert solution.goodDaysToRobBank([33, 72, 90, 36, 50, 51], 44) == []
assert solution.goodDaysToRobBank([56, 86, 63, 49, 83], 40) == []
assert solution.goodDaysToRobBank([50, 49, 95, 31, 14, 44, 72, 38], 75) == []
assert solution.goodDaysToRobBank([8, 67, 70, 6, 87, 73, 7, 48, 45], 49) == []
assert solution.goodDaysToRobBank([52, 34, 90, 42, 25, 30, 93], 59) == []
assert solution.goodDaysToRobBank([91, 89], 40) == []
assert solution.goodDaysToRobBank([68, 57, 8, 77, 39, 71, 22, 12, 11, 83], 71) == []
assert solution.goodDaysToRobBank([34, 76, 64, 10, 24, 87, 50, 74, 81, 91], 104) == []
assert solution.goodDaysToRobBank([97, 79, 31], 94) == []
assert solution.goodDaysToRobBank([74, 81, 70, 50, 41, 8, 89, 88, 21, 33], 31) == []
assert solution.goodDaysToRobBank([8, 11, 86, 72, 79, 96], 91) == []
assert solution.goodDaysToRobBank([26, 77, 79, 10, 30, 31, 1, 29], 57) == []
assert solution.goodDaysToRobBank([77, 4, 43, 96, 9, 72, 24, 85, 21], 37) == []
assert solution.goodDaysToRobBank([84, 61, 82, 45, 21, 70, 83, 92], 174) == []
assert solution.goodDaysToRobBank([24, 12, 96, 39, 92, 34, 33, 82, 16], 181) == []
assert solution.goodDaysToRobBank([82, 90, 97, 48, 29, 42, 16], 188) == []
assert solution.goodDaysToRobBank([34, 20, 8, 18, 14, 97], 34) == []
assert solution.goodDaysToRobBank([64, 16, 50, 51, 69], 20) == []
assert solution.goodDaysToRobBank([51, 48, 65, 9, 90], 184) == []
assert solution.goodDaysToRobBank([6, 99, 15, 41], 83) == []
assert solution.goodDaysToRobBank([39, 28, 22, 26, 89, 38], 199) == []
assert solution.goodDaysToRobBank([35, 68, 33, 67], 191) == []
assert solution.goodDaysToRobBank([98, 48, 39, 87, 21, 79], 38) == []
assert solution.goodDaysToRobBank([32, 71, 64, 24, 41, 7, 100, 27, 84], 52) == []
assert solution.goodDaysToRobBank([77, 28, 5, 88], 169) == []
assert solution.goodDaysToRobBank([76, 94, 2, 88, 54, 56], 29) == []
assert solution.goodDaysToRobBank([70, 40, 39, 21, 93, 91, 3, 58, 23], 128) == []
assert solution.goodDaysToRobBank([28, 84, 30], 81) == []
assert solution.goodDaysToRobBank([77, 9], 83) == []
assert solution.goodDaysToRobBank([46, 13, 3], 92) == []
assert solution.goodDaysToRobBank([91, 86, 52, 93, 69], 164) == []
assert solution.goodDaysToRobBank([51, 4], 8) == []
assert solution.goodDaysToRobBank([21, 68], 114) == []
assert solution.goodDaysToRobBank([74, 46, 13, 64, 9, 26], 98) == []
assert solution.goodDaysToRobBank([63, 95, 65, 26, 42, 90, 77, 33, 50], 36) == []
assert solution.goodDaysToRobBank([95, 59, 57, 98, 13, 76], 185) == []
assert solution.goodDaysToRobBank([43, 87, 46, 68, 83, 74, 97, 17], 78) == []
assert solution.goodDaysToRobBank([37, 35, 17, 86, 99, 52, 92, 28], 17) == []
assert solution.goodDaysToRobBank([17, 22, 90, 92, 69], 91) == []
assert solution.goodDaysToRobBank([17, 10], 163) == []
assert solution.goodDaysToRobBank([38, 18, 51, 63], 129) == []
assert solution.goodDaysToRobBank([85, 21, 32, 42, 66, 96, 27, 78], 30) == []
assert solution.goodDaysToRobBank([9, 36, 27, 29, 96, 32, 75, 38, 90, 91], 151) == []
assert solution.goodDaysToRobBank([97, 57, 32], 161) == []
assert solution.goodDaysToRobBank([91, 99, 9, 72, 84, 88], 161) == []
assert solution.goodDaysToRobBank([71, 92, 76, 12, 26], 86) == []
assert solution.goodDaysToRobBank([48, 74, 43], 35) == []
assert solution.goodDaysToRobBank([8, 16, 66, 13, 74, 34, 9], 57) == []
assert solution.goodDaysToRobBank([57, 91], 121) == []
assert solution.goodDaysToRobBank([98, 54, 77, 34, 44, 57], 180) == []
assert solution.goodDaysToRobBank([53, 24, 21, 81, 32, 46, 30, 72, 98], 178) == []
assert solution.goodDaysToRobBank([16, 9, 59, 93, 32, 52, 71], 41) == []
assert solution.goodDaysToRobBank([31, 40, 54], 161) == []
assert solution.goodDaysToRobBank([30, 34, 6, 64, 55, 63], 186) == []
assert solution.goodDaysToRobBank([46, 73, 59, 2, 51], 93) == []
assert solution.goodDaysToRobBank([73, 83, 79], 105) == []
assert solution.goodDaysToRobBank([15, 96, 78], 124) == []
assert solution.goodDaysToRobBank([57, 68, 63, 37, 56, 95], 190) == []
assert solution.goodDaysToRobBank([66, 57, 26, 86, 17, 94, 31, 72, 61, 71], 51) == []
assert solution.goodDaysToRobBank([23, 15, 37, 1, 2, 95], 5) == []
assert solution.goodDaysToRobBank([58, 51, 31, 17, 82, 48, 42, 46, 78], 195) == []
assert solution.goodDaysToRobBank([84, 3, 7, 10, 59, 73, 72, 20, 86], 101) == []
assert solution.goodDaysToRobBank([64, 33, 25, 97], 58) == []
assert solution.goodDaysToRobBank([1, 68], 97) == []
assert solution.goodDaysToRobBank([16, 88, 81, 40], 160) == []
assert solution.goodDaysToRobBank([51, 14, 40, 63, 5, 45, 74, 61], 16) == []
assert solution.goodDaysToRobBank([6, 61, 59, 65, 35, 26, 20, 94, 85, 100], 40) == []
assert solution.goodDaysToRobBank([15, 66, 59, 4], 63) == []
assert solution.goodDaysToRobBank([97, 90, 21], 127) == []
assert solution.goodDaysToRobBank([38, 44, 32, 67, 1], 111) == []
assert solution.goodDaysToRobBank([15, 24, 100, 81, 65], 91) == []
assert solution.goodDaysToRobBank([20, 45, 5, 84, 17, 14, 25, 32, 69, 1], 137) == []
assert solution.goodDaysToRobBank([51, 88, 17, 77, 26, 23], 198) == []
assert solution.goodDaysToRobBank([98, 24, 1, 90, 81, 44, 17, 96], 104) == []
assert solution.goodDaysToRobBank([89, 78, 83], 2) == []
assert solution.goodDaysToRobBank([47, 60, 67, 82, 91, 10, 6, 100, 22, 65], 178) == []
assert solution.goodDaysToRobBank([78, 15, 79, 26, 31, 69, 34, 40, 95], 9) == []
assert solution.goodDaysToRobBank([94, 47, 22], 55) == []
assert solution.goodDaysToRobBank([28, 56, 92, 95, 22, 43, 89, 47], 75) == []
assert solution.goodDaysToRobBank([51, 53, 90, 8, 37], 193) == []
assert solution.goodDaysToRobBank([89, 73, 49, 94, 74], 161) == []
assert solution.goodDaysToRobBank([21, 36, 78, 96, 4, 82, 11], 81) == []
assert solution.goodDaysToRobBank([18, 34, 71, 98, 14, 86, 3], 24) == []
assert solution.goodDaysToRobBank([64, 97, 61, 55, 53, 48, 7], 46) == []
assert solution.goodDaysToRobBank([67, 75, 70], 154) == []
assert solution.goodDaysToRobBank([33, 59, 24, 6, 78, 36, 89, 28, 50, 86], 26) == []