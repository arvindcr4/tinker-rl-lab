
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
    def equalizeWater(self, buckets: List[int], loss: int) -> float:
        """
        We want to find the maximum equal amount x such that we can redistribute water
        from buckets with more than x to buckets with less than x, accounting for loss.
        
        For a given target x, we can check if it's feasible:
        - Buckets with water > x can contribute (buckets[i] - x) gallons.
        - But when we pour k gallons, only k * (1 - loss/100) arrives at the destination.
        - So the effective amount received from a source is (buckets[i] - x) * (1 - loss/100).
        - Buckets with water < x need (x - buckets[i]) gallons.
        
        The total amount that can be effectively transferred must be >= total amount needed.
        
        Let factor = 1 - loss / 100. This is the fraction that arrives.
        
        For a target x:
        surplus = sum(max(0, buckets[i] - x) for i)
        deficit = sum(max(0, x - buckets[i]) for i)
        
        The effective surplus available is surplus * factor.
        We need: surplus * factor >= deficit
        
        This function is monotonic in x: if x is feasible, any x' < x is also feasible.
        So we can binary search on x.
        
        Lower bound: min(buckets) (we can always achieve at least the minimum)
        Upper bound: sum(buckets) / len(buckets) (the average, but due to loss, we can't exceed this)
        Actually, the upper bound could be sum(buckets) / (len(buckets) * factor) but that might be too high.
        A safe upper bound is max(buckets) or even sum(buckets).
        
        Let's use binary search with sufficient iterations.
        """
        n = len(buckets)
        total = sum(buckets)
        
        # Lower bound: 0, but practically min(buckets)
        # Upper bound: total (if loss=0, answer is total/n; with loss, answer <= total/n)
        # Actually, with loss, the maximum possible equal amount is at most total / n (if loss=0)
        # With loss > 0, it's less. But let's use a safe upper bound.
        
        lo = 0.0
        hi = total  # safe upper bound
        
        factor = 1.0 - loss / 100.0
        
        # Binary search for 100 iterations for precision
        for _ in range(100):
            mid = (lo + hi) / 2.0
            
            # Check if mid is feasible
            surplus = 0.0
            deficit = 0.0
            for b in buckets:
                if b > mid:
                    surplus += b - mid
                else:
                    deficit += mid - b
            
            # Effective amount that can be transferred
            if surplus * factor >= deficit:
                lo = mid
            else:
                hi = mid
        
        return lo

solution=Solution()
assert solution.equalizeWater([100, 95, 66, 46], 30) == 73.08823466300964
assert solution.equalizeWater([50, 14, 17, 29, 24, 71], 45) == 29.519606113433838
assert solution.equalizeWater([4, 9, 26, 15], 54) == 10.91095781326294
assert solution.equalizeWater([82, 53, 39], 76) == 48.24323844909668
assert solution.equalizeWater([41, 77, 55, 28, 95, 36, 64, 76, 49, 33], 93) == 36.95988088846207
assert solution.equalizeWater([35, 61], 24) == 46.22726786136627
assert solution.equalizeWater([43, 76, 54, 61, 20], 92) == 29.333330154418945
assert solution.equalizeWater([76, 68, 7, 45], 32) == 44.578941345214844
assert solution.equalizeWater([29, 28, 7, 2, 9, 32], 85) == 9.086952209472656
assert solution.equalizeWater([64, 57, 58, 18, 66, 53, 10, 15, 80, 17], 35) == 38.69619369506836
assert solution.equalizeWater([76, 79, 50, 47, 4, 42, 77, 97, 91, 71], 60) == 53.031247198581696
assert solution.equalizeWater([2, 63, 23, 45, 58, 97, 40, 60, 22, 81], 35) == 44.25316375494003
assert solution.equalizeWater([17, 89, 52, 56], 70) == 40.05263012647629
assert solution.equalizeWater([67, 93, 3, 91, 70, 63, 79, 38], 58) == 52.09291595220566
assert solution.equalizeWater([83, 79, 82, 41, 28, 50, 61, 87], 17) == 62.121582090854645
assert solution.equalizeWater([95, 90, 83, 55, 21, 48, 2, 82, 13, 81], 76) == 35.07692098617554
assert solution.equalizeWater([4, 34, 51, 89], 18) == 41.97801744937897
assert solution.equalizeWater([42], 86) == 41.99999499320984
assert solution.equalizeWater([39, 66, 53, 98, 42, 35, 25], 31) == 47.896207213401794
assert solution.equalizeWater([1, 24], 26) == 10.781604766845703
assert solution.equalizeWater([44, 6, 85, 40, 81, 22, 45, 75, 55, 59], 85) == 31.437496542930603
assert solution.equalizeWater([27], 51) == 26.999993562698364
assert solution.equalizeWater([46, 85, 13, 10, 56, 44, 75, 1], 96) == 10.607140958309174
assert solution.equalizeWater([20, 48, 11, 29], 76) == 19.93022918701172
assert solution.equalizeWater([52, 38, 40, 44, 43, 76], 72) == 44.04385423660278
assert solution.equalizeWater([62, 25, 14, 98, 26, 39, 43, 12, 46], 93) == 19.971882343292236
assert solution.equalizeWater([59, 54, 40, 18, 86, 98, 44], 78) == 42.9096736907959
assert solution.equalizeWater([99, 44, 51, 52, 36, 93, 35, 90], 14) == 60.75461554527283
assert solution.equalizeWater([51, 7, 45, 58, 92], 10) == 49.55318760871887
assert solution.equalizeWater([100, 69], 49) == 79.47019338607788
assert solution.equalizeWater([85, 19, 91, 71, 97, 48, 73, 62], 88) == 44.695650815963745
assert solution.equalizeWater([93, 68, 40, 92, 21, 72], 81) == 44.47463607788086
assert solution.equalizeWater([1, 89, 8, 69, 2, 82], 8) == 40.2430517077446
assert solution.equalizeWater([81, 73, 85, 69], 88) == 71.82352811098099
assert solution.equalizeWater([85, 83, 13, 72], 12) == 61.59340292215347
assert solution.equalizeWater([98, 4, 20, 87, 54, 27], 10) == 46.684205174446106
assert solution.equalizeWater([33, 23], 44) == 26.589738607406616
assert solution.equalizeWater([22, 23, 67, 97], 4) == 51.64285320043564
assert solution.equalizeWater([68, 38, 6, 59, 69, 17, 73], 4) == 46.672510623931885
assert solution.equalizeWater([94, 9, 21, 83, 40, 79, 15, 17], 68) == 29.681816935539246
assert solution.equalizeWater([89, 92, 86, 55, 62, 15, 32, 11, 95, 85], 66) == 46.423790752887726
assert solution.equalizeWater([91, 65, 57, 77, 29], 46) == 58.51380878686905
assert solution.equalizeWater([39, 11, 96, 43, 12, 72, 40, 33, 31], 15) == 40.391807556152344
assert solution.equalizeWater([39, 53, 79, 7, 48, 25, 33], 40) == 36.370365142822266
assert solution.equalizeWater([21], 44) == 20.99999499320984
assert solution.equalizeWater([42], 33) == 41.99999499320984
assert solution.equalizeWater([4], 25) == 3.9999923706054688
assert solution.equalizeWater([21, 22, 80, 97, 84], 2) == 60.48177695274353
assert solution.equalizeWater([26, 63, 85, 83, 11], 46) == 44.6795579791069
assert solution.equalizeWater([90, 41, 8, 84, 6, 89, 44], 90) == 19.51999604701996
assert solution.equalizeWater([96, 4, 10, 11, 19, 29, 9], 80) == 13.652172088623047
assert solution.equalizeWater([14, 93, 40, 10, 83, 4, 65], 14) == 41.83282595872879
assert solution.equalizeWater([53, 76, 82, 100], 92) == 59.38709378242493
assert solution.equalizeWater([22, 51, 3, 31, 73, 71, 74, 82], 71) == 35.45842170715332
assert solution.equalizeWater([73, 5, 36, 17, 44, 80, 20], 60) == 29.391298294067383
assert solution.equalizeWater([51, 58, 73], 59) == 57.5329624414444
assert solution.equalizeWater([91, 4, 17, 29, 31, 74, 5, 8, 26, 22], 83) == 15.823387444019318
assert solution.equalizeWater([11, 9, 13, 50, 46, 38, 52], 58) == 23.743587970733643
assert solution.equalizeWater([100, 76], 58) == 83.09859037399292
assert solution.equalizeWater([90, 88], 53) == 88.63945484161377
assert solution.equalizeWater([58, 17, 41, 91, 8, 84, 16, 83, 52, 79], 72) == 35.81451255083084
assert solution.equalizeWater([28, 21], 16) == 24.19564723968506
assert solution.equalizeWater([47], 9) == 46.99999439716339
assert solution.equalizeWater([81, 39], 45) == 53.90322482585907
assert solution.equalizeWater([30, 61, 20, 73, 13, 99, 60, 29, 23, 36], 53) == 36.63832068443298
assert solution.equalizeWater([96, 29, 66, 53], 37) == 56.46011924743652
assert solution.equalizeWater([93, 16, 28], 32) == 40.01492303609848
assert solution.equalizeWater([41, 8, 25], 55) == 19.84210228919983
assert solution.equalizeWater([36, 61, 37, 13, 41, 95, 45, 15], 48) == 37.30920821428299
assert solution.equalizeWater([81, 87, 10, 11, 3, 73, 31, 78, 15, 2], 4) == 38.43902152776718
assert solution.equalizeWater([83, 7, 77], 84) == 24.69696354866028
assert solution.equalizeWater([23, 77, 21, 26, 91, 17, 46], 75) == 29.57894217967987
assert solution.equalizeWater([72, 43, 96, 61, 71, 50], 38) == 62.17695236206055
assert solution.equalizeWater([48, 41, 79, 54, 27, 12, 43, 19, 82], 65) == 35.18627142906189
assert solution.equalizeWater([36, 39, 43, 23, 11, 91, 62, 4], 83) == 21.516555547714233
assert solution.equalizeWater([19, 70, 61, 67, 30, 46, 5, 48, 99, 55], 38) == 45.0299688577652
assert solution.equalizeWater([32, 2, 21, 30, 13, 22, 34], 90) == 10.749998092651367
assert solution.equalizeWater([46], 10) == 45.99999451637268
assert solution.equalizeWater([5, 78, 21, 7], 92) == 9.222217798233032
assert solution.equalizeWater([17, 21, 19, 92], 7) == 36.27480888366699
assert solution.equalizeWater([80, 62, 33, 13, 87, 61], 68) == 42.31707036495209
assert solution.equalizeWater([55, 3, 82, 17, 57, 27, 12, 94, 37, 70], 38) == 39.25431931018829
assert solution.equalizeWater([62, 9], 9) == 34.251305103302
assert solution.equalizeWater([42, 97, 36, 35, 94], 76) == 45.643673837184906
assert solution.equalizeWater([39, 56, 60, 11], 71) == 29.91978406906128
assert solution.equalizeWater([64, 14, 79, 78, 46, 57, 71, 82, 62, 49], 34) == 57.00501894950867
assert solution.equalizeWater([87, 76, 68, 91, 18, 17, 90, 95, 5, 28], 48) == 46.57864719629288
assert solution.equalizeWater([84, 66, 25, 9, 48], 86) == 25.504128456115723
assert solution.equalizeWater([88, 9, 87, 7, 84, 37, 25], 30) == 42.50819444656372
assert solution.equalizeWater([3, 33, 90, 13, 51, 38, 64], 75) == 26.153844594955444
assert solution.equalizeWater([86, 23, 2, 58, 80, 27, 85, 9], 1) == 46.09421730041504
assert solution.equalizeWater([62], 88) == 61.99999260902405
assert solution.equalizeWater([51, 95, 84], 16) == 75.13432711362839
assert solution.equalizeWater([86, 57, 59, 39], 25) == 58.49999713897705
assert solution.equalizeWater([24, 32, 14], 25) == 22.399993896484375
assert solution.equalizeWater([30, 46, 60, 66], 15) == 49.486483097076416
assert solution.equalizeWater([86], 73) == 85.99999487400055
assert solution.equalizeWater([62, 55, 18, 33], 40) == 37.87499713897705
assert solution.equalizeWater([89, 37, 94], 21) == 70.37596440315247
assert solution.equalizeWater([72, 7, 71], 60) == 35.666659355163574