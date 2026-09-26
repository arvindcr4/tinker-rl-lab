
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
    def findFinalValue(self, nums: List[int], original: int) -> int:
        s = set(nums)
        while original in s:
            original <<= 1
        return original

solution=Solution()
assert solution.findFinalValue([17, 7, 100, 30, 69, 38, 93], 802) == 802
assert solution.findFinalValue([21, 34, 32, 59, 11, 2, 92, 25], 195) == 195
assert solution.findFinalValue([39, 87, 55, 95, 71, 91], 72) == 72
assert solution.findFinalValue([36, 57, 66, 67, 96, 64, 13, 86, 43, 34], 687) == 687
assert solution.findFinalValue([99, 34, 92, 67, 8, 61, 20, 35, 14], 427) == 427
assert solution.findFinalValue([71, 50], 47) == 47
assert solution.findFinalValue([78, 77, 3, 5, 21, 80], 161) == 161
assert solution.findFinalValue([47, 16, 53], 376) == 376
assert solution.findFinalValue([22, 58, 86, 8, 17, 23, 21], 741) == 741
assert solution.findFinalValue([64, 89, 65], 246) == 246
assert solution.findFinalValue([75, 70, 44, 95, 33, 3, 71, 31, 38], 23) == 23
assert solution.findFinalValue([1, 9, 95, 32, 60, 12, 74, 29], 174) == 174
assert solution.findFinalValue([61, 8, 87, 78, 33, 79, 9, 93, 27], 104) == 104
assert solution.findFinalValue([92, 22], 384) == 384
assert solution.findFinalValue([29, 93, 53, 68, 88, 87, 81], 443) == 443
assert solution.findFinalValue([64, 63, 95, 26], 865) == 865
assert solution.findFinalValue([19, 29, 45, 56, 77, 10, 23, 93, 91], 6) == 6
assert solution.findFinalValue([29, 80, 11, 47, 74, 13, 86], 413) == 413
assert solution.findFinalValue([78, 48, 5, 93, 47, 57, 25, 6], 150) == 150
assert solution.findFinalValue([96, 82, 39, 31, 16, 36, 89, 56, 87, 46], 986) == 986
assert solution.findFinalValue([38, 76], 554) == 554
assert solution.findFinalValue([4, 29], 264) == 264
assert solution.findFinalValue([66, 91, 99, 86, 42, 20, 29, 62], 555) == 555
assert solution.findFinalValue([56, 62], 962) == 962
assert solution.findFinalValue([85, 46], 882) == 882
assert solution.findFinalValue([8, 63, 48, 73, 5, 54, 71, 90], 183) == 183
assert solution.findFinalValue([50, 79, 15, 20], 936) == 936
assert solution.findFinalValue([82, 92, 5, 98, 90, 64, 23, 74, 10], 381) == 381
assert solution.findFinalValue([5, 85], 227) == 227
assert solution.findFinalValue([69, 22, 41, 91, 3, 100, 95, 71, 34], 702) == 702
assert solution.findFinalValue([90, 72, 12, 25, 49, 71, 16, 39], 382) == 382
assert solution.findFinalValue([58, 76, 11, 98, 3, 6, 14, 62, 100], 133) == 133
assert solution.findFinalValue([68, 32, 79, 71, 3, 67, 39, 81, 33], 527) == 527
assert solution.findFinalValue([63, 66, 18, 76, 58, 98], 312) == 312
assert solution.findFinalValue([70, 66, 4, 13, 15], 82) == 82
assert solution.findFinalValue([36, 10], 287) == 287
assert solution.findFinalValue([58, 83, 65, 52], 405) == 405
assert solution.findFinalValue([48, 73, 3, 75, 81, 24, 50, 96], 561) == 561
assert solution.findFinalValue([41, 94, 2, 85, 36, 39, 1], 755) == 755
assert solution.findFinalValue([64, 95, 63, 92], 350) == 350
assert solution.findFinalValue([45, 14, 69, 88, 63, 28, 21], 878) == 878
assert solution.findFinalValue([34, 61, 1, 43, 11, 31, 85, 12, 26], 668) == 668
assert solution.findFinalValue([10, 71, 79], 271) == 271
assert solution.findFinalValue([42, 68, 65, 26, 50, 34, 6, 73, 45, 69], 515) == 515
assert solution.findFinalValue([62, 95], 298) == 298
assert solution.findFinalValue([21, 22, 42, 60], 401) == 401
assert solution.findFinalValue([87, 57, 32, 18, 15], 114) == 114
assert solution.findFinalValue([5, 89, 73, 82, 100, 55], 428) == 428
assert solution.findFinalValue([76, 41, 43, 37, 61, 38, 42, 9, 49], 699) == 699
assert solution.findFinalValue([56, 91, 46, 60, 86, 2, 10, 37], 905) == 905
assert solution.findFinalValue([39, 21, 40, 10, 60, 29, 57], 730) == 730
assert solution.findFinalValue([12, 5, 90, 46, 83, 50], 283) == 283
assert solution.findFinalValue([86, 68], 607) == 607
assert solution.findFinalValue([5, 4, 45, 81, 15, 79, 1], 526) == 526
assert solution.findFinalValue([98, 9, 53, 78, 1], 961) == 961
assert solution.findFinalValue([48, 68, 73, 84, 4, 86, 52, 10, 8, 35], 567) == 567
assert solution.findFinalValue([91, 60, 87, 10, 80, 29, 25, 27], 811) == 811
assert solution.findFinalValue([33, 53, 81, 96], 946) == 946
assert solution.findFinalValue([45, 83, 85], 337) == 337
assert solution.findFinalValue([48, 29], 485) == 485
assert solution.findFinalValue([43, 17, 89, 80, 84, 81, 91, 96, 25], 350) == 350
assert solution.findFinalValue([4, 95, 69, 48, 96, 12, 67, 94], 304) == 304
assert solution.findFinalValue([7, 69, 82], 558) == 558
assert solution.findFinalValue([40, 41, 16, 60, 55], 579) == 579
assert solution.findFinalValue([12, 8, 38, 49, 46, 24, 57, 31, 58], 998) == 998
assert solution.findFinalValue([77, 27, 29, 76, 21, 87, 42, 65, 43, 57], 918) == 918
assert solution.findFinalValue([55, 14, 96, 98, 36], 645) == 645
assert solution.findFinalValue([57, 83], 718) == 718
assert solution.findFinalValue([79, 30, 54, 75], 968) == 968
assert solution.findFinalValue([97, 91, 12, 64, 82, 5], 596) == 596
assert solution.findFinalValue([56, 99, 30, 37, 7, 93], 741) == 741
assert solution.findFinalValue([53, 2, 92, 96, 87, 30, 56, 34, 36], 940) == 940
assert solution.findFinalValue([46, 14, 56, 44, 92, 58, 77], 319) == 319
assert solution.findFinalValue([38, 33, 45, 28, 30, 2], 621) == 621
assert solution.findFinalValue([17, 39, 46, 41, 18], 108) == 108
assert solution.findFinalValue([6, 45, 91, 89], 264) == 264
assert solution.findFinalValue([59, 81, 1], 238) == 238
assert solution.findFinalValue([15, 26], 281) == 281
assert solution.findFinalValue([39, 25, 47, 6], 641) == 641
assert solution.findFinalValue([62, 16, 32, 40, 77, 90, 93, 94, 23], 267) == 267
assert solution.findFinalValue([45, 62, 36, 91, 49, 37, 3], 84) == 84
assert solution.findFinalValue([43, 20, 66, 61, 75, 25], 927) == 927
assert solution.findFinalValue([58, 94, 38, 31], 939) == 939
assert solution.findFinalValue([55, 8, 68, 47, 15, 6, 2, 78, 10], 812) == 812
assert solution.findFinalValue([26, 11], 400) == 400
assert solution.findFinalValue([17, 43, 88, 13, 99, 75, 57, 20, 48], 150) == 150
assert solution.findFinalValue([54, 91, 7, 46], 788) == 788
assert solution.findFinalValue([82, 36, 13, 53, 65, 64, 98, 14, 68], 460) == 460
assert solution.findFinalValue([61, 69, 90, 37, 81, 83], 904) == 904
assert solution.findFinalValue([46, 70, 27], 227) == 227
assert solution.findFinalValue([77, 25], 81) == 81
assert solution.findFinalValue([53, 43, 25], 886) == 886
assert solution.findFinalValue([96, 88, 25, 66, 26], 422) == 422
assert solution.findFinalValue([13, 71, 48, 1, 55], 920) == 920
assert solution.findFinalValue([1, 3, 34, 38, 14, 12, 76, 32, 23], 734) == 734
assert solution.findFinalValue([50, 76, 27, 41, 94, 51, 68, 73, 33, 23], 143) == 143
assert solution.findFinalValue([80, 28, 40, 95, 3, 46, 10, 68, 78], 78) == 156
assert solution.findFinalValue([67, 88, 21], 852) == 852
assert solution.findFinalValue([66, 81, 52, 83], 401) == 401
assert solution.findFinalValue([21, 27, 83, 56], 513) == 513