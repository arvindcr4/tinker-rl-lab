
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
    def isAdditiveNumber(self, num: str) -> bool:
        """
        Determine if a string is an additive number.
        An additive number is a string whose digits can form an additive sequence.
        A valid additive sequence should contain at least three numbers.
        Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two.
        Numbers in the additive sequence cannot have leading zeros.
        """
        n = len(num)
        
        def is_valid_sequence(first: int, second: int, start: int) -> bool:
            """
            Check if the remaining string starting from 'start' forms a valid additive sequence
            given the first two numbers are num[first:second] and num[second:start].
            """
            # We already have at least two numbers, now we need at least one more to have a sequence of 3+
            # Actually, the function should verify that the rest of the string can be decomposed into
            # numbers where each is the sum of the previous two.
            
            prev1 = int(num[first:second])
            prev2 = int(num[second:start])
            
            # Now continue from position 'start'
            i = start
            while i < n:
                # The next number should be prev1 + prev2
                next_val = prev1 + prev2
                next_str = str(next_val)
                next_len = len(next_str)
                
                # Check if the remaining string starts with next_str
                if i + next_len > n:
                    return False
                if num[i:i+next_len] != next_str:
                    return False
                
                # Check for leading zero: if next_val is 0 but the string has more than one digit, or if next_val > 0 but starts with '0'
                # Actually, since we're converting to string and comparing, if next_val is 0, next_str is "0", which is fine.
                # If next_val > 0, next_str won't start with '0'. So this is already handled by the string comparison.
                # But we need to ensure that the substring doesn't have a leading zero unless it's just "0".
                # Since next_str is the canonical string representation, if num[i:i+next_len] == next_str, then it's valid.
                # However, we must also ensure that the substring we're taking doesn't have extra leading zeros.
                # For example, if next_val is 5, next_str is "5". If num[i] is '0' and next_len is 1, that's fine (it's "0" only if next_val is 0).
                # Actually, the check num[i:i+next_len] != next_str already handles this because "05" != "5".
                
                prev1 = prev2
                prev2 = next_val
                i += next_len
            
            # If we've consumed the entire string, it's valid
            return i == n
        
        # Try all possible first and second numbers
        # First number: num[0:i] for i in range(1, n-1) (need at least 2 more digits for second number and third number)
        # Second number: num[i:j] for j in range(i+1, n)
        
        for i in range(1, n - 1):
            # First number: num[0:i]
            # Check for leading zero
            if num[0] == '0' and i > 1:
                break  # Can't have leading zeros, so if first digit is 0, first number must be just "0"
            
            first_str = num[0:i]
            # If first number has leading zero and length > 1, invalid
            if len(first_str) > 1 and first_str[0] == '0':
                break
            
            for j in range(i + 1, n):
                # Second number: num[i:j]
                second_str = num[i:j]
                if len(second_str) > 1 and second_str[0] == '0':
                    break  # Can't have leading zeros for second number if length > 1
                
                # Now check if the rest of the string forms a valid additive sequence
                if is_valid_sequence(0, i, j):
                    return True
        
        return False

solution=Solution()
assert solution.isAdditiveNumber('72011') == False
assert solution.isAdditiveNumber('21568') == False
assert solution.isAdditiveNumber('81235') == False
assert solution.isAdditiveNumber('61677') == False
assert solution.isAdditiveNumber('43362') == False
assert solution.isAdditiveNumber('45736') == False
assert solution.isAdditiveNumber('13371') == False
assert solution.isAdditiveNumber('51352') == False
assert solution.isAdditiveNumber('65349') == False
assert solution.isAdditiveNumber('6320') == False
assert solution.isAdditiveNumber('43098') == False
assert solution.isAdditiveNumber('19181') == False
assert solution.isAdditiveNumber('58549') == False
assert solution.isAdditiveNumber('7915') == False
assert solution.isAdditiveNumber('12919') == False
assert solution.isAdditiveNumber('82134') == False
assert solution.isAdditiveNumber('44898') == False
assert solution.isAdditiveNumber('56947') == False
assert solution.isAdditiveNumber('19548') == False
assert solution.isAdditiveNumber('9960') == False
assert solution.isAdditiveNumber('17640') == False
assert solution.isAdditiveNumber('6498') == False
assert solution.isAdditiveNumber('71565') == False
assert solution.isAdditiveNumber('53742') == True
assert solution.isAdditiveNumber('91402') == False
assert solution.isAdditiveNumber('96277') == False
assert solution.isAdditiveNumber('47309') == False
assert solution.isAdditiveNumber('63292') == False
assert solution.isAdditiveNumber('72337') == False
assert solution.isAdditiveNumber('80724') == False
assert solution.isAdditiveNumber('78862') == False
assert solution.isAdditiveNumber('59920') == False
assert solution.isAdditiveNumber('61115') == False
assert solution.isAdditiveNumber('46690') == False
assert solution.isAdditiveNumber('70958') == False
assert solution.isAdditiveNumber('14264') == False
assert solution.isAdditiveNumber('89129') == False
assert solution.isAdditiveNumber('42047') == False
assert solution.isAdditiveNumber('28011') == False
assert solution.isAdditiveNumber('35893') == False
assert solution.isAdditiveNumber('9607') == False
assert solution.isAdditiveNumber('45132') == False
assert solution.isAdditiveNumber('37596') == False
assert solution.isAdditiveNumber('43894') == False
assert solution.isAdditiveNumber('3438') == False
assert solution.isAdditiveNumber('533') == False
assert solution.isAdditiveNumber('84401') == False
assert solution.isAdditiveNumber('20862') == False
assert solution.isAdditiveNumber('25911') == False
assert solution.isAdditiveNumber('66420') == False
assert solution.isAdditiveNumber('89753') == False
assert solution.isAdditiveNumber('78747') == False
assert solution.isAdditiveNumber('21965') == False
assert solution.isAdditiveNumber('17213') == False
assert solution.isAdditiveNumber('19101') == False
assert solution.isAdditiveNumber('42963') == False
assert solution.isAdditiveNumber('40539') == False
assert solution.isAdditiveNumber('43906') == False
assert solution.isAdditiveNumber('60077') == False
assert solution.isAdditiveNumber('62870') == True
assert solution.isAdditiveNumber('78592') == True
assert solution.isAdditiveNumber('69407') == False
assert solution.isAdditiveNumber('94014') == False
assert solution.isAdditiveNumber('43308') == False
assert solution.isAdditiveNumber('44694') == False
assert solution.isAdditiveNumber('56199') == False
assert solution.isAdditiveNumber('30124') == False
assert solution.isAdditiveNumber('58329') == False
assert solution.isAdditiveNumber('14688') == False
assert solution.isAdditiveNumber('23304') == False
assert solution.isAdditiveNumber('1084') == False
assert solution.isAdditiveNumber('61018') == False
assert solution.isAdditiveNumber('12911') == False
assert solution.isAdditiveNumber('86246') == False
assert solution.isAdditiveNumber('59217') == False
assert solution.isAdditiveNumber('99809') == False
assert solution.isAdditiveNumber('35380') == False
assert solution.isAdditiveNumber('26306') == False
assert solution.isAdditiveNumber('41844') == False
assert solution.isAdditiveNumber('41428') == False
assert solution.isAdditiveNumber('75735') == False
assert solution.isAdditiveNumber('39642') == False
assert solution.isAdditiveNumber('85494') == False
assert solution.isAdditiveNumber('1639') == False
assert solution.isAdditiveNumber('52809') == False
assert solution.isAdditiveNumber('43589') == False
assert solution.isAdditiveNumber('2246') == True
assert solution.isAdditiveNumber('21866') == False
assert solution.isAdditiveNumber('81011') == False
assert solution.isAdditiveNumber('35403') == False
assert solution.isAdditiveNumber('64604') == False
assert solution.isAdditiveNumber('91413') == False
assert solution.isAdditiveNumber('55941') == False
assert solution.isAdditiveNumber('65181') == False
assert solution.isAdditiveNumber('90403') == False
assert solution.isAdditiveNumber('34931') == False
assert solution.isAdditiveNumber('32663') == False
assert solution.isAdditiveNumber('65283') == False
assert solution.isAdditiveNumber('96188') == False
assert solution.isAdditiveNumber('41270') == False