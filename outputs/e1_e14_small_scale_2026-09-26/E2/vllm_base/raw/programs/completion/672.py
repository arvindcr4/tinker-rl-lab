
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
    def flipLights(self, n: int, presses: int) -> int:
        # If no presses, only one state (all on)
        if presses == 0:
            return 1
        
        # If n == 1, there are only 2 possible states: on or off
        if n == 1:
            return 2
        
        # If n == 2, let's analyze:
        # Button 1: flips both
        # Button 2: flips bulb 2
        # Button 3: flips bulb 1
        # Button 4: flips bulb 1 (since 1 = 3*0+1, 4 > 2 so only bulb 1)
        # So buttons 1, 3, 4 affect bulb 1; buttons 1, 2 affect bulb 2
        # With presses >= 1, we can get multiple states
        if n == 2:
            if presses == 1:
                return 3
            else:
                return 4
        
        # For n >= 3, the pattern repeats every 6 bulbs, but actually the key insight is:
        # The state of each bulb depends on which buttons were pressed an odd number of times.
        # There are 4 buttons, so 2^4 = 16 possible combinations of button press parities.
        # However, not all combinations are reachable with exactly `presses` presses.
        # Also, bulbs beyond a certain point have redundant patterns.
        
        # Key observation: The behavior of bulbs repeats every 6 bulbs, but for counting distinct states,
        # we only need to consider the first few bulbs because the patterns are periodic.
        # Actually, it's known that for n >= 3, the number of distinct states depends on presses.
        
        # Let's think about it differently. The state of the system is determined by which buttons
        # are pressed an odd number of times. Let b1, b2, b3, b4 be 0 or 1 indicating if button i
        # was pressed an odd number of times. The total number of presses must have the same parity
        # as b1 + b2 + b3 + b4 (since each press changes the parity of one button).
        
        # For n >= 3, the distinct states are determined by the first 3 bulbs (or even fewer).
        # It's a known result that:
        # - If presses == 0: 1 state
        # - If presses == 1: 4 states (for n >= 3)
        # - If presses == 2: 7 states (for n >= 3)
        # - If presses >= 3: 8 states (for n >= 3)
        
        # But let's verify with the examples and constraints.
        
        # For n >= 3:
        if n >= 3:
            if presses == 1:
                return 4
            elif presses == 2:
                return 7
            else:
                return 8
        
        return 0

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