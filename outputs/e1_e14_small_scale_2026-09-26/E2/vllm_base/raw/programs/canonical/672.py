
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
    def flipLights(self, n: int, presses: int) -> int:
        ops = (0b111111, 0b010101, 0b101010, 0b100100)
        n = min(n, 6)
        vis = set()
        for mask in range(1 << 4):
            cnt = mask.bit_count()
            if cnt <= presses and cnt % 2 == presses % 2:
                t = 0
                for i, op in enumerate(ops):
                    if (mask >> i) & 1:
                        t ^= op
                t &= (1 << 6) - 1
                t >>= 6 - n
                vis.add(t)
        return len(vis)

solution=Solution()
assert solution.flipLights(16, 656) == 8
assert solution.flipLights(575, 57) == 8
assert solution.flipLights(780, 334) == 8
assert solution.flipLights(718, 616) == 8
assert solution.flipLights(334, 969) == 8
assert solution.flipLights(856, 839) == 8
assert solution.flipLights(590, 712) == 8
assert solution.flipLights(541, 405) == 8
assert solution.flipLights(972, 994) == 8
assert solution.flipLights(908, 559) == 8
assert solution.flipLights(166, 563) == 8
assert solution.flipLights(157, 253) == 8
assert solution.flipLights(400, 543) == 8
assert solution.flipLights(807, 27) == 8
assert solution.flipLights(794, 802) == 8
assert solution.flipLights(885, 922) == 8
assert solution.flipLights(711, 783) == 8
assert solution.flipLights(230, 640) == 8
assert solution.flipLights(873, 628) == 8
assert solution.flipLights(578, 738) == 8
assert solution.flipLights(41, 968) == 8
assert solution.flipLights(863, 334) == 8
assert solution.flipLights(714, 528) == 8
assert solution.flipLights(854, 210) == 8
assert solution.flipLights(478, 468) == 8
assert solution.flipLights(654, 219) == 8
assert solution.flipLights(637, 778) == 8
assert solution.flipLights(725, 958) == 8
assert solution.flipLights(431, 116) == 8
assert solution.flipLights(767, 417) == 8
assert solution.flipLights(131, 494) == 8
assert solution.flipLights(929, 142) == 8
assert solution.flipLights(473, 628) == 8
assert solution.flipLights(35, 415) == 8
assert solution.flipLights(129, 311) == 8
assert solution.flipLights(438, 704) == 8
assert solution.flipLights(960, 678) == 8
assert solution.flipLights(869, 343) == 8
assert solution.flipLights(25, 639) == 8
assert solution.flipLights(808, 946) == 8
assert solution.flipLights(136, 255) == 8
assert solution.flipLights(246, 667) == 8
assert solution.flipLights(40, 466) == 8
assert solution.flipLights(735, 513) == 8
assert solution.flipLights(699, 394) == 8
assert solution.flipLights(636, 426) == 8
assert solution.flipLights(955, 636) == 8
assert solution.flipLights(352, 191) == 8
assert solution.flipLights(983, 859) == 8
assert solution.flipLights(6, 892) == 8
assert solution.flipLights(238, 193) == 8
assert solution.flipLights(81, 92) == 8
assert solution.flipLights(687, 839) == 8
assert solution.flipLights(666, 287) == 8
assert solution.flipLights(345, 495) == 8
assert solution.flipLights(291, 831) == 8
assert solution.flipLights(488, 421) == 8
assert solution.flipLights(924, 863) == 8
assert solution.flipLights(469, 116) == 8
assert solution.flipLights(863, 844) == 8
assert solution.flipLights(121, 473) == 8
assert solution.flipLights(164, 753) == 8
assert solution.flipLights(434, 629) == 8
assert solution.flipLights(196, 653) == 8
assert solution.flipLights(115, 311) == 8
assert solution.flipLights(128, 423) == 8
assert solution.flipLights(569, 908) == 8
assert solution.flipLights(500, 823) == 8
assert solution.flipLights(39, 257) == 8
assert solution.flipLights(465, 450) == 8
assert solution.flipLights(925, 218) == 8
assert solution.flipLights(565, 939) == 8
assert solution.flipLights(100, 736) == 8
assert solution.flipLights(182, 946) == 8
assert solution.flipLights(91, 893) == 8
assert solution.flipLights(596, 271) == 8
assert solution.flipLights(572, 448) == 8
assert solution.flipLights(605, 806) == 8
assert solution.flipLights(482, 5) == 8
assert solution.flipLights(411, 570) == 8
assert solution.flipLights(615, 810) == 8
assert solution.flipLights(289, 628) == 8
assert solution.flipLights(913, 176) == 8
assert solution.flipLights(443, 351) == 8
assert solution.flipLights(465, 95) == 8
assert solution.flipLights(75, 629) == 8
assert solution.flipLights(315, 169) == 8
assert solution.flipLights(715, 933) == 8
assert solution.flipLights(429, 142) == 8
assert solution.flipLights(385, 38) == 8
assert solution.flipLights(781, 510) == 8
assert solution.flipLights(276, 512) == 8
assert solution.flipLights(228, 424) == 8
assert solution.flipLights(323, 300) == 8
assert solution.flipLights(110, 654) == 8
assert solution.flipLights(516, 459) == 8
assert solution.flipLights(703, 876) == 8
assert solution.flipLights(932, 804) == 8
assert solution.flipLights(704, 230) == 8
assert solution.flipLights(646, 530) == 8