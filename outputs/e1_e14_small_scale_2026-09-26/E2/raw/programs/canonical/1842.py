
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
    def nextPalindrome(self, num: str) -> str:
        def next_permutation(nums: List[str]) -> bool:
            n = len(nums) // 2
            i = n - 2
            while i >= 0 and nums[i] >= nums[i + 1]:
                i -= 1
            if i < 0:
                return False
            j = n - 1
            while j >= 0 and nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
            nums[i + 1 : n] = nums[i + 1 : n][::-1]
            return True

        nums = list(num)
        if not next_permutation(nums):
            return ""
        n = len(nums)
        for i in range(n // 2):
            nums[n - i - 1] = nums[i]
        return "".join(nums)

solution=Solution()
assert solution.nextPalindrome(5228668225) == 5262882625
assert solution.nextPalindrome(2948778492) == 2974884792
assert solution.nextPalindrome(6119779116) == 6171991716
assert solution.nextPalindrome(4294994924) == 4299449924
assert solution.nextPalindrome(9684664869) == 9686446869
assert solution.nextPalindrome(6135885316) == 6138558316
assert solution.nextPalindrome(8123553218) == 8125335218
assert solution.nextPalindrome(6093443906) == 6094334906
assert solution.nextPalindrome(5912552195) == 5915225195
assert solution.nextPalindrome(8947887498) == 8948778498
assert solution.nextPalindrome(4189009814) == 4190880914
assert solution.nextPalindrome(4671661764) == 4676116764
assert solution.nextPalindrome(1464444641) == 1644444461
assert solution.nextPalindrome(4681111864) == 4811661184
assert solution.nextPalindrome(79599597) == 79955997
assert solution.nextPalindrome(1086226801) == 1206886021
assert solution.nextPalindrome(6577997756) == 6579779756
assert solution.nextPalindrome(5773003775) == 7035775307
assert solution.nextPalindrome(4699559964) == 4956996594
assert solution.nextPalindrome(1806776081) == 1807667081
assert solution.nextPalindrome(7161881617) == 7168118617
assert solution.nextPalindrome(9915665199) == 9916556199
assert solution.nextPalindrome(4094114904) == 4104994014
assert solution.nextPalindrome(8675885768) == 8678558768
assert solution.nextPalindrome(4631771364) == 4637117364
assert solution.nextPalindrome(9679669769) == 9696776969
assert solution.nextPalindrome(7680990867) == 7689009867
assert solution.nextPalindrome(2007447002) == 2040770402
assert solution.nextPalindrome(5484444845) == 5844444485
assert solution.nextPalindrome(8731441378) == 8734114378
assert solution.nextPalindrome(5308668035) == 5360880635
assert solution.nextPalindrome(8918008198) == 8980110898
assert solution.nextPalindrome(9454664549) == 9456446549
assert solution.nextPalindrome(8545995458) == 8549559458
assert solution.nextPalindrome(24588542) == 24855842
assert solution.nextPalindrome(4815555184) == 4851551584
assert solution.nextPalindrome(5176666715) == 5616776165
assert solution.nextPalindrome(5890770985) == 5897007985
assert solution.nextPalindrome(8604554068) == 8605445068
assert solution.nextPalindrome(5875445785) == 7455885547
assert solution.nextPalindrome(1955665591) == 1956556591
assert solution.nextPalindrome(9789559879) == 9795885979
assert solution.nextPalindrome(8426006248) == 8460220648
assert solution.nextPalindrome(1408118041) == 1410880141
assert solution.nextPalindrome(2709449072) == 2740990472
assert solution.nextPalindrome(4193553914) == 4195335914
assert solution.nextPalindrome(8827447288) == 8842772488
assert solution.nextPalindrome(2948448492) == 2984444892
assert solution.nextPalindrome(3886556883) == 5368888635
assert solution.nextPalindrome(7817667187) == 7861771687
assert solution.nextPalindrome(7847337487) == 7873443787
assert solution.nextPalindrome(5167007615) == 5170660715
assert solution.nextPalindrome(9820220289) == 9822002289
assert solution.nextPalindrome(94133149) == 94311349
assert solution.nextPalindrome(2476116742) == 2614774162
assert solution.nextPalindrome(9940770499) == 9947007499
assert solution.nextPalindrome(27677672) == 27766772
assert solution.nextPalindrome(6209009026) == 6290000926
assert solution.nextPalindrome(3237997323) == 3239779323
assert solution.nextPalindrome(2684884862) == 2688448862
assert solution.nextPalindrome(8983113898) == 9138888319
assert solution.nextPalindrome(4013003104) == 4030110304
assert solution.nextPalindrome(1122332211) == 1123223211
assert solution.nextPalindrome(1516776151) == 1517667151
assert solution.nextPalindrome(5837337385) == 5873333785
assert solution.nextPalindrome(7191441917) == 7194114917
assert solution.nextPalindrome(4139669314) == 4163993614
assert solution.nextPalindrome(5748228475) == 5782442875
assert solution.nextPalindrome(3886776883) == 3887667883
assert solution.nextPalindrome(5646886465) == 5648668465
assert solution.nextPalindrome(8486776848) == 8487667848
assert solution.nextPalindrome(1147777411) == 1174774711
assert solution.nextPalindrome(3895665983) == 3896556983
assert solution.nextPalindrome(3548668453) == 3564884653
assert solution.nextPalindrome(5955885595) == 5958558595
assert solution.nextPalindrome(9526446259) == 9542662459
assert solution.nextPalindrome(3172552713) == 3175225713
assert solution.nextPalindrome(2711441172) == 2714114172
assert solution.nextPalindrome(3841991483) == 3849119483
assert solution.nextPalindrome(9496446949) == 9644994469
assert solution.nextPalindrome(64388346) == 64833846
assert solution.nextPalindrome(4042552404) == 4045225404
assert solution.nextPalindrome(8391881938) == 8398118938
assert solution.nextPalindrome(9558228559) == 9582552859
assert solution.nextPalindrome(7711441177) == 7714114177
assert solution.nextPalindrome(1590000951) == 1900550091
assert solution.nextPalindrome(8976666798) == 9667887669
assert solution.nextPalindrome(1503993051) == 1509339051
assert solution.nextPalindrome(4163883614) == 4168338614
assert solution.nextPalindrome(7686336867) == 7836666387
assert solution.nextPalindrome(5658338565) == 5683553865
assert solution.nextPalindrome(1551881551) == 1558118551
assert solution.nextPalindrome(1492882941) == 1498228941
assert solution.nextPalindrome(1945885491) == 1948558491
assert solution.nextPalindrome(94811849) == 98144189
assert solution.nextPalindrome(50166105) == 50611605
assert solution.nextPalindrome(8539669358) == 8563993658
assert solution.nextPalindrome(4787337874) == 4837777384