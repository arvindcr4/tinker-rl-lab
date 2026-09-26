
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
    def maximumXOR(self, nums: List[int]) -> int:
        # The operation: nums[i] = nums[i] AND (nums[i] XOR x)
        # Let's analyze what this operation does.
        # nums[i] AND (nums[i] XOR x)
        # For each bit position:
        # - If nums[i] has bit 0, then nums[i] AND anything will have bit 0.
        # - If nums[i] has bit 1, then nums[i] XOR x has bit (1 XOR x_bit), and then AND with 1 gives (1 XOR x_bit).
        #   So we can choose x_bit to be 0 or 1, meaning we can set the bit to 1 or 0.
        #
        # Wait, let me re-analyze:
        # For a bit position b:
        # - If nums[i][b] = 0: result[b] = 0 AND (0 XOR x[b]) = 0 AND x[b] = 0. So the bit stays 0.
        # - If nums[i][b] = 1: result[b] = 1 AND (1 XOR x[b]) = 1 AND (1 if x[b]=0, 0 if x[b]=1) = 1 if x[b]=0, 0 if x[b]=1.
        #   So we can choose to keep it as 1 or change it to 0.
        #
        # So the operation allows us to change any bit that is 1 to either 1 or 0. Bits that are 0 must stay 0.
        # In other words, for each element, we can only turn 1-bits into 0-bits, but never turn 0-bits into 1-bits.
        # This means each element can only decrease (in terms of set bits) or stay the same.
        #
        # We want to maximize the XOR of all elements.
        # 
        # Key insight: The XOR of all elements can be at most the OR of all elements (since we can only remove bits, not add them).
        # But more precisely, we need to think about what values we can achieve.
        #
        # Actually, let's think about it differently. For each bit position, can we achieve a 1 in the final XOR?
        # 
        # Let's consider each bit position independently... but XOR doesn't work independently per bit in a simple way for maximization.
        #
        # Alternative approach: 
        # Since we can only turn 1s to 0s in each number, the set of possible values for each nums[i] is a subset of its bits.
        # We want to choose subsets to maximize the XOR.
        #
        # This is related to the concept of linear basis in GF(2).
        # 
        # Actually, let me think again. The key observation from similar problems:
        # The maximum XOR we can achieve is the XOR of all elements if we don't apply any operation... but we might do better by applying operations.
        #
        # Wait, let's look at the examples:
        # Example 1: [3,2,4,6] -> 3=011, 2=010, 4=100, 6=110
        # XOR = 3^2^4^6 = 011^010=001, 001^100=101, 101^110=011 = 3. But output is 7.
        # After operation: [3,2,4,2] -> 3^2^4^2 = 011^010=001, 001^100=101, 101^010=111 = 7.
        #
        # So by changing 6 (110) to 2 (010), we removed the bit at position 2 (value 4) from that element.
        #
        # The insight is: we can use Gaussian elimination / linear basis to find the maximum XOR.
        # 
        # Actually, the maximum possible XOR from a set of numbers where each number can be replaced by any submask (subset of bits) is equal to the XOR basis maximum.
        #
        # More precisely, the answer is the maximum value achievable by XORing a subset of the numbers, where each number can be any submask of the original.
        #
        # But here's the key: if we have a linear basis for the original numbers, the maximum XOR we can get is the same as the maximum XOR from the basis.
        # And since we can only remove bits, the span (in GF(2)) of all possible submasks is the same as the span of the original numbers.
        #
        # So the answer is the maximum XOR achievable using a linear basis of the original array.
        
        # Build a linear basis and find the maximum XOR
        basis = []
        
        for num in nums:
            for b in basis:
                num = min(num, num ^ b)
            if num > 0:
                # Insert into basis, maintaining the property that each basis element has a unique highest set bit
                j = len(basis)
                while j > 0 and num > basis[j - 1]:
                    j -= 1
                basis.insert(j, num)
                # Reduce new basis element with existing ones and vice versa
                for k in range(len(basis)):
                    if k != j:
                        basis[k] = min(basis[k], basis[k] ^ num)
                        if basis[k] == 0:
                            basis.pop(k)
                            break
        
        # Now compute maximum XOR from the basis
        result = 0
        for b in basis:
            result = max(result, result ^ b)
        
        return result

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