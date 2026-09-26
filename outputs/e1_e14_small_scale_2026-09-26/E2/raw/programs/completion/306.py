
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
        n = len(num)
        
        def is_valid_additive_sequence(first: int, second: int, start: int) -> bool:
            """
            Check if the rest of the string starting from 'start' forms a valid additive sequence
            given the first two numbers are num[first:second] and num[second:start].
            """
            prev = num[first:second]
            curr = num[second:start]
            
            # Check for leading zeros
            if len(prev) > 1 and prev[0] == '0':
                return False
            if len(curr) > 1 and curr[0] == '0':
                return False
            
            prev_num = int(prev)
            curr_num = int(curr)
            
            idx = start
            while idx < n:
                next_num = prev_num + curr_num
                next_str = str(next_num)
                next_len = len(next_str)
                
                if idx + next_len > n:
                    return False
                
                if num[idx:idx + next_len] != next_str:
                    return False
                
                prev_num = curr_num
                curr_num = next_num
                idx += next_len
            
            return True
        
        # Try all possible first and second numbers
        # First number: num[0:i]
        # Second number: num[i:j]
        # i ranges from 1 to n-2 (at least one digit for first, at least one for second, and at least one more for third)
        # j ranges from i+1 to n-1 (at least one digit for second, and remaining for rest)
        
        for i in range(1, n - 1):
            # First number: num[0:i]
            first_str = num[0:i]
            # Check leading zero for first number
            if len(first_str) > 1 and first_str[0] == '0':
                break  # No need to continue with longer first numbers since they'll also have leading zero
            
            for j in range(i + 1, n):
                # Second number: num[i:j]
                second_str = num[i:j]
                # Check leading zero for second number
                if len(second_str) > 1 and second_str[0] == '0':
                    break  # No need to continue with longer second numbers since they'll also have leading zero
                
                if is_valid_additive_sequence(0, i, j):
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