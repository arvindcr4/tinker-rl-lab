
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
from collections import Counter

class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        # Count the digits of n
        n_counter = Counter(str(n))
        
        # Check all powers of 2 up to 10^9
        # 2^0 = 1, 2^29 = 536870912, 2^30 = 1073741824 > 10^9
        # So we need to check 2^0 through 2^29
        for i in range(31):  # 2^0 to 2^30, but 2^30 > 10^9, so up to 2^29 is enough, but let's be safe
            power_of_2 = 1 << i
            if power_of_2 > 10**9:
                break
            if Counter(str(power_of_2)) == n_counter:
                return True
        return False

solution=Solution()
assert solution.reorderedPowerOf2(800102732) == False
assert solution.reorderedPowerOf2(301883004) == False
assert solution.reorderedPowerOf2(533848910) == False
assert solution.reorderedPowerOf2(770781565) == False
assert solution.reorderedPowerOf2(22991246) == False
assert solution.reorderedPowerOf2(356691298) == False
assert solution.reorderedPowerOf2(738612481) == False
assert solution.reorderedPowerOf2(837867096) == False
assert solution.reorderedPowerOf2(652956354) == False
assert solution.reorderedPowerOf2(917482948) == False
assert solution.reorderedPowerOf2(532239386) == False
assert solution.reorderedPowerOf2(676080000) == False
assert solution.reorderedPowerOf2(577122358) == False
assert solution.reorderedPowerOf2(787145695) == False
assert solution.reorderedPowerOf2(198090588) == False
assert solution.reorderedPowerOf2(539735528) == False
assert solution.reorderedPowerOf2(56189278) == False
assert solution.reorderedPowerOf2(563345508) == False
assert solution.reorderedPowerOf2(637790575) == False
assert solution.reorderedPowerOf2(651874479) == False
assert solution.reorderedPowerOf2(705907456) == False
assert solution.reorderedPowerOf2(719664511) == False
assert solution.reorderedPowerOf2(455453033) == False
assert solution.reorderedPowerOf2(207925267) == False
assert solution.reorderedPowerOf2(538542699) == False
assert solution.reorderedPowerOf2(532389761) == False
assert solution.reorderedPowerOf2(195802495) == False
assert solution.reorderedPowerOf2(15779807) == False
assert solution.reorderedPowerOf2(544642136) == False
assert solution.reorderedPowerOf2(419051292) == False
assert solution.reorderedPowerOf2(116800715) == False
assert solution.reorderedPowerOf2(561614207) == False
assert solution.reorderedPowerOf2(182966494) == False
assert solution.reorderedPowerOf2(405553305) == False
assert solution.reorderedPowerOf2(109263419) == False
assert solution.reorderedPowerOf2(243932085) == False
assert solution.reorderedPowerOf2(787543976) == False
assert solution.reorderedPowerOf2(332871466) == False
assert solution.reorderedPowerOf2(498269713) == False
assert solution.reorderedPowerOf2(200878451) == False
assert solution.reorderedPowerOf2(406378683) == False
assert solution.reorderedPowerOf2(432436590) == False
assert solution.reorderedPowerOf2(986752240) == False
assert solution.reorderedPowerOf2(911877870) == False
assert solution.reorderedPowerOf2(115935877) == False
assert solution.reorderedPowerOf2(771094784) == False
assert solution.reorderedPowerOf2(500559026) == False
assert solution.reorderedPowerOf2(884631187) == False
assert solution.reorderedPowerOf2(506728556) == False
assert solution.reorderedPowerOf2(684650093) == False
assert solution.reorderedPowerOf2(726941618) == False
assert solution.reorderedPowerOf2(834857883) == False
assert solution.reorderedPowerOf2(470942435) == False
assert solution.reorderedPowerOf2(925851620) == False
assert solution.reorderedPowerOf2(110003053) == False
assert solution.reorderedPowerOf2(794788068) == False
assert solution.reorderedPowerOf2(793198267) == False
assert solution.reorderedPowerOf2(496614715) == False
assert solution.reorderedPowerOf2(19487806) == False
assert solution.reorderedPowerOf2(524398665) == False
assert solution.reorderedPowerOf2(953164421) == False
assert solution.reorderedPowerOf2(109964052) == False
assert solution.reorderedPowerOf2(651205154) == False
assert solution.reorderedPowerOf2(394012323) == False
assert solution.reorderedPowerOf2(945318822) == False
assert solution.reorderedPowerOf2(603942023) == False
assert solution.reorderedPowerOf2(194051542) == False
assert solution.reorderedPowerOf2(39334228) == False
assert solution.reorderedPowerOf2(310635196) == False
assert solution.reorderedPowerOf2(864534030) == False
assert solution.reorderedPowerOf2(643525420) == False
assert solution.reorderedPowerOf2(627545237) == False
assert solution.reorderedPowerOf2(91761416) == False
assert solution.reorderedPowerOf2(775794005) == False
assert solution.reorderedPowerOf2(797520042) == False
assert solution.reorderedPowerOf2(869568843) == False
assert solution.reorderedPowerOf2(745216826) == False
assert solution.reorderedPowerOf2(627125788) == False
assert solution.reorderedPowerOf2(564817808) == False
assert solution.reorderedPowerOf2(16210273) == False
assert solution.reorderedPowerOf2(553576203) == False
assert solution.reorderedPowerOf2(1391387) == False
assert solution.reorderedPowerOf2(50500196) == False
assert solution.reorderedPowerOf2(840265405) == False
assert solution.reorderedPowerOf2(998808769) == False
assert solution.reorderedPowerOf2(639430469) == False
assert solution.reorderedPowerOf2(491141967) == False
assert solution.reorderedPowerOf2(903213307) == False
assert solution.reorderedPowerOf2(833525631) == False
assert solution.reorderedPowerOf2(221163693) == False
assert solution.reorderedPowerOf2(893297516) == False
assert solution.reorderedPowerOf2(405921945) == False
assert solution.reorderedPowerOf2(373282584) == False
assert solution.reorderedPowerOf2(867681832) == False
assert solution.reorderedPowerOf2(18237021) == False
assert solution.reorderedPowerOf2(70413164) == False
assert solution.reorderedPowerOf2(243224145) == False
assert solution.reorderedPowerOf2(948191441) == False
assert solution.reorderedPowerOf2(469648804) == False
assert solution.reorderedPowerOf2(193830521) == False