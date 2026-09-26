
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
from collections import deque

class Solution:
    def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i] represents the maximum sum of a non-empty subsequence ending at index i
        # such that for every two consecutive integers in the subsequence, the distance is <= k.
        # dp[i] = nums[i] + max(0, max(dp[j]) for j in [i-k, i-1])
        
        # We use a deque to maintain the maximum of dp values in the sliding window of size k.
        # The deque will store indices j such that dp[j] is decreasing.
        
        dp = [0] * n
        dp[0] = nums[0]
        max_sum = dp[0]
        
        # Deque stores indices, and we maintain it such that dp[deque[0]] is the maximum in the window
        dq = deque()
        dq.append(0)
        
        for i in range(1, n):
            # Remove indices that are out of the window [i-k, i-1]
            while dq and dq[0] < i - k:
                dq.popleft()
            
            # The best previous dp value is at dq[0]
            best_prev = dp[dq[0]]
            
            # dp[i] = nums[i] + max(0, best_prev)
            dp[i] = nums[i] + max(0, best_prev)
            
            # Maintain the deque in decreasing order of dp values
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            dq.append(i)
            
            max_sum = max(max_sum, dp[i])
        
        return max_sum

solution=Solution()
assert solution.constrainedSubsetSum([8973, -3696, -2096, 1850, 5286, -7459, 406, 158], 4) == 16673
assert solution.constrainedSubsetSum([-9133, 3224, -5768, 1018, -7354, -7539, -8639], 6) == 4242
assert solution.constrainedSubsetSum([-9332, 1810, 2491, 8836, 1854, 5661, -6729, 2472], 4) == 23124
assert solution.constrainedSubsetSum([596, 7663, 8066, 1019, 4461, 9004], 6) == 30809
assert solution.constrainedSubsetSum([-4250, 1259, -9386, -427, -6919, -4327, -4735], 1) == 1259
assert solution.constrainedSubsetSum([-5740], 1) == -5740
assert solution.constrainedSubsetSum([-6896, -564, 2767, -4157, 7195, -3400, 1660, -1334, 954], 6) == 12576
assert solution.constrainedSubsetSum([-111, -4924, -9923, -2504], 3) == -111
assert solution.constrainedSubsetSum([5393, 7126, 5934, 8274, 2034, -4803, 2617, -5442], 6) == 31378
assert solution.constrainedSubsetSum([9295, 4481, -2692, 4175, 1227, 4723, -6179, 4993, -1796], 6) == 28894
assert solution.constrainedSubsetSum([897, 5862, -4299], 2) == 6759
assert solution.constrainedSubsetSum([1589, 633, -6352], 1) == 2222
assert solution.constrainedSubsetSum([9471, -8769, -9289, 7952, -7509, 7835, -6429, -7152], 5) == 25258
assert solution.constrainedSubsetSum([-5724, 1206, -8483], 1) == 1206
assert solution.constrainedSubsetSum([-2708, 3398, -3067, -4371, 610], 3) == 4008
assert solution.constrainedSubsetSum([7797, 274, 5976, 7386, 3272, 8346, -3937, -5836], 6) == 33051
assert solution.constrainedSubsetSum([8041, 7864, -4485, -6273, 8703, -3055, 7162, 9408], 8) == 41178
assert solution.constrainedSubsetSum([5991, 9076, 3138, -1106, -8037, -492, 9000, -9205, -5973, 5026], 6) == 32231
assert solution.constrainedSubsetSum([-3625, -1841, -3980, 8947, -564, -1568, -1836], 5) == 8947
assert solution.constrainedSubsetSum([-3133, -8263, 6561, -1210], 4) == 6561
assert solution.constrainedSubsetSum([-4130, -3699, -6868, -7115, -2717, 5283, 156, -6377, 8041, -9232], 7) == 13480
assert solution.constrainedSubsetSum([-6712], 1) == -6712
assert solution.constrainedSubsetSum([-252, -3773, -3836], 3) == -252
assert solution.constrainedSubsetSum([3578, -7051, 3861, -2291, -8456, 6692, -8730, 7438], 5) == 21569
assert solution.constrainedSubsetSum([6751, -70, 5099, 8683, 2875, -611, 9443, 8560, -1297, -4728], 10) == 41411
assert solution.constrainedSubsetSum([-6703, -3941, 8550, 3282], 3) == 11832
assert solution.constrainedSubsetSum([-2213, 7204, 5806, -169, -5986, -5826, -8078], 4) == 13010
assert solution.constrainedSubsetSum([-9276, 3796, 4546], 3) == 8342
assert solution.constrainedSubsetSum([-1218, -5098, 4040, 3817, 1699, -6134, -8865, -6613], 5) == 9556
assert solution.constrainedSubsetSum([2764, 5227, 9922, -6778, -5081, 1651], 5) == 19564
assert solution.constrainedSubsetSum([2888], 1) == 2888
assert solution.constrainedSubsetSum([-8809, 7514, 4650, -2123], 3) == 12164
assert solution.constrainedSubsetSum([-49, 2941, -1150, 1622, 5373, -779, 780, -2645, 7392], 9) == 18108
assert solution.constrainedSubsetSum([2472, 8947, -8754, 8846], 4) == 20265
assert solution.constrainedSubsetSum([-44, -9042, 3664, -7977, 1645], 1) == 3664
assert solution.constrainedSubsetSum([-6109, 7837], 2) == 7837
assert solution.constrainedSubsetSum([-2423, -4610, 4710, -9295, 9140], 3) == 13850
assert solution.constrainedSubsetSum([-1273, 706], 2) == 706
assert solution.constrainedSubsetSum([3616, -6254, -6997, -2849, 440, -5553], 3) == 3616
assert solution.constrainedSubsetSum([-5789, -5959, -8649, -2295, -8516, 6582, -2023, -6143, 9407, 2886], 1) == 12293
assert solution.constrainedSubsetSum([6285, -7495, 9499, 6794, -6500, 9767], 4) == 32345
assert solution.constrainedSubsetSum([6267, -2865, 6304, 5853, -4607, -2692, 6735, 5155], 1) == 20150
assert solution.constrainedSubsetSum([-9748, -9415, 4594], 1) == 4594
assert solution.constrainedSubsetSum([-3852, -5973, 8824, 1215, 8220], 1) == 18259
assert solution.constrainedSubsetSum([7643, 8357, -1657], 3) == 16000
assert solution.constrainedSubsetSum([7099, -1803, 851, -2022, 3740, -5904, -5597, 7544, -5785], 2) == 13637
assert solution.constrainedSubsetSum([2959], 1) == 2959
assert solution.constrainedSubsetSum([9082, -2207, 5870], 2) == 14952
assert solution.constrainedSubsetSum([-4423, -2823], 1) == -2823
assert solution.constrainedSubsetSum([9993, -9125], 1) == 9993
assert solution.constrainedSubsetSum([-2362, 2379, 1137, -2779, 1070, 8727, -8417], 5) == 13313
assert solution.constrainedSubsetSum([3364, -7673, -8039, -6728, 8419, 2194, 3801], 1) == 14414
assert solution.constrainedSubsetSum([1542], 1) == 1542
assert solution.constrainedSubsetSum([-7805, -8354, -5714, 7920, 1352, 6422, -9457, 9076, 3979, -8422], 2) == 28749
assert solution.constrainedSubsetSum([8990, 4965, -6911, 4623, -9827, 452, -4587], 3) == 19030
assert solution.constrainedSubsetSum([5196, -1870, -2028], 1) == 5196
assert solution.constrainedSubsetSum([3754, -896, 8723, 8471, -981, 7429, -816, -2356], 5) == 28377
assert solution.constrainedSubsetSum([3282, 1903, 2676, 2534, -1837, 4455, 7548, 9602, 3246], 6) == 35246
assert solution.constrainedSubsetSum([4860, 7891, 419, -3386], 2) == 13170
assert solution.constrainedSubsetSum([-1699, -1446, 1870, -9198, -6266, -8940, -9876, -5123], 4) == 1870
assert solution.constrainedSubsetSum([-9395, 4844, 972, -9285, -4950], 4) == 5816
assert solution.constrainedSubsetSum([-6292], 1) == -6292
assert solution.constrainedSubsetSum([2433, -8135, -1767, 2148], 4) == 4581
assert solution.constrainedSubsetSum([-5913, -4678, 2709, -5145, -3274, -3507, -847, -6170], 6) == 2709
assert solution.constrainedSubsetSum([3674, 9832, -3083, 5883, -3062, 3872, 9978, -263], 2) == 33239
assert solution.constrainedSubsetSum([1876, -8440, 5360, -2861], 2) == 7236
assert solution.constrainedSubsetSum([-4092, 6970, -4347, 8254, 7917, 6602], 5) == 29743
assert solution.constrainedSubsetSum([-6611, 1416, -7104, 7502, -2578], 3) == 8918
assert solution.constrainedSubsetSum([6807], 1) == 6807
assert solution.constrainedSubsetSum([8281, -936, -7090, -8104, 649, 5393, 9816], 4) == 24139
assert solution.constrainedSubsetSum([-7920, 1231, 898, 6592], 4) == 8721
assert solution.constrainedSubsetSum([2059, -9489], 2) == 2059
assert solution.constrainedSubsetSum([7915, 4685, -4665, -5078, 7867, -6549, 9501, -5719, 2771, -7547], 5) == 32739
assert solution.constrainedSubsetSum([3122, 5143, 9449, 5572, 4960], 4) == 28246
assert solution.constrainedSubsetSum([-1299, -8766, -3460, 6910, -9290, -4182, -4985, 229, 9698, 1395], 7) == 18232
assert solution.constrainedSubsetSum([5225, 5095, 8316, 8115, -8733, 8349, -211], 1) == 26751
assert solution.constrainedSubsetSum([-8482, -7166, 6937, 4523, -6878], 1) == 11460
assert solution.constrainedSubsetSum([-6533, 9354, 9201, -9044, -8620, 7778], 1) == 18555
assert solution.constrainedSubsetSum([-7112, 6149, 5549, -4681, -4172, 6857, 2570, 9413, -3597, -8548], 10) == 30538
assert solution.constrainedSubsetSum([-8074], 1) == -8074
assert solution.constrainedSubsetSum([-3255, -2258, 5090, -4398, 4349, -8102], 4) == 9439
assert solution.constrainedSubsetSum([-6670, -2345, -1884, 2062, 7285, -3064, 1462], 3) == 10809
assert solution.constrainedSubsetSum([5099, -7309, -8569, 9638, 2855, -3426, 2829, -1022, -8571, 5601], 6) == 26022
assert solution.constrainedSubsetSum([-5167, -6061, -7640, -9926, -2066, -3321, 1707, 9448, -2089, -6394], 2) == 11155
assert solution.constrainedSubsetSum([-1187, 2860, -1915, -2686, -310, 7335, 9782, 6163, -285], 5) == 26140
assert solution.constrainedSubsetSum([444, 2595, -4169, -3610, -1381, -7538, -3119], 5) == 3039
assert solution.constrainedSubsetSum([4536, -2875, -7460, 9949, -7937, -8024, -2252, 2880, -5400, -1549], 4) == 17365
assert solution.constrainedSubsetSum([1154, -5482, -1822, 899, -3917, 8434, 6893, 8237, -8760], 5) == 25617
assert solution.constrainedSubsetSum([1112, -4369, -1493, 8842, -5000, 7099], 4) == 17053
assert solution.constrainedSubsetSum([2758, -8606, -3737, 1737, 643, -6450, -4965], 5) == 5138
assert solution.constrainedSubsetSum([-9186, -8960], 2) == -8960
assert solution.constrainedSubsetSum([-6995], 1) == -6995
assert solution.constrainedSubsetSum([-3793, 2587, -3954, 744, 6227, 4445], 4) == 14003
assert solution.constrainedSubsetSum([-7535, 6142, -8189, 2367, -6412, 4579, 7978], 1) == 12557
assert solution.constrainedSubsetSum([-9227, -8365], 1) == -8365
assert solution.constrainedSubsetSum([9996, -9919, -9410, 5774, 8596, 1289, -874, -4009], 3) == 25655
assert solution.constrainedSubsetSum([1921, -311], 1) == 1921
assert solution.constrainedSubsetSum([567, 8006, 1886, 2884, -7919, -3327], 5) == 13343
assert solution.constrainedSubsetSum([-7597, -3287], 2) == -3287
assert solution.constrainedSubsetSum([1600, 6538, 9600], 3) == 17738