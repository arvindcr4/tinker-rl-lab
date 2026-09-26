
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
    def makeStringsEqual(self, s: str, target: str) -> bool:
        return ("1" in s) == ("1" in target)

solution=Solution()
assert solution.makeStringsEqual('11101', '00100') == True
assert solution.makeStringsEqual('0111', '1010') == True
assert solution.makeStringsEqual('010111001', '111000000') == True
assert solution.makeStringsEqual('101101001', '011111101') == True
assert solution.makeStringsEqual('10110', '01000') == True
assert solution.makeStringsEqual('01', '00') == False
assert solution.makeStringsEqual('10', '01') == True
assert solution.makeStringsEqual('110100100', '001110101') == True
assert solution.makeStringsEqual('0010', '1011') == True
assert solution.makeStringsEqual('0100101000', '0101101011') == True
assert solution.makeStringsEqual('1000', '1010') == True
assert solution.makeStringsEqual('001', '101') == True
assert solution.makeStringsEqual('001011101', '111100100') == True
assert solution.makeStringsEqual('1010010010', '1111000001') == True
assert solution.makeStringsEqual('1110000100', '1011101011') == True
assert solution.makeStringsEqual('1110111111', '0101010011') == True
assert solution.makeStringsEqual('00111', '01000') == True
assert solution.makeStringsEqual('11101', '01011') == True
assert solution.makeStringsEqual('11', '00') == False
assert solution.makeStringsEqual('1010', '1101') == True
assert solution.makeStringsEqual('00101000', '10101000') == True
assert solution.makeStringsEqual('11000100', '10100101') == True
assert solution.makeStringsEqual('00011111', '00011111') == True
assert solution.makeStringsEqual('0101110000', '1110001101') == True
assert solution.makeStringsEqual('1100001010', '0011111110') == True
assert solution.makeStringsEqual('100010', '110101') == True
assert solution.makeStringsEqual('0001', '0101') == True
assert solution.makeStringsEqual('010', '110') == True
assert solution.makeStringsEqual('0010', '1001') == True
assert solution.makeStringsEqual('1010011100', '0001011101') == True
assert solution.makeStringsEqual('0000', '0110') == False
assert solution.makeStringsEqual('01000000', '10011001') == True
assert solution.makeStringsEqual('100100', '111000') == True
assert solution.makeStringsEqual('110011', '110110') == True
assert solution.makeStringsEqual('11101111', '11100111') == True
assert solution.makeStringsEqual('01100110', '00110111') == True
assert solution.makeStringsEqual('01100', '10110') == True
assert solution.makeStringsEqual('001', '100') == True
assert solution.makeStringsEqual('1101111111', '0010101010') == True
assert solution.makeStringsEqual('1100011', '0010000') == True
assert solution.makeStringsEqual('0000', '0011') == False
assert solution.makeStringsEqual('1011000110', '0101011111') == True
assert solution.makeStringsEqual('00', '11') == False
assert solution.makeStringsEqual('10111', '00011') == True
assert solution.makeStringsEqual('01001', '01010') == True
assert solution.makeStringsEqual('100001', '010001') == True
assert solution.makeStringsEqual('01010111', '00110101') == True
assert solution.makeStringsEqual('00000', '00111') == False
assert solution.makeStringsEqual('11000111', '11101010') == True
assert solution.makeStringsEqual('01100110', '11011100') == True
assert solution.makeStringsEqual('100', '010') == True
assert solution.makeStringsEqual('1011', '1001') == True
assert solution.makeStringsEqual('111001', '000101') == True
assert solution.makeStringsEqual('0000100101', '1010101000') == True
assert solution.makeStringsEqual('001100', '110100') == True
assert solution.makeStringsEqual('001110011', '000010101') == True
assert solution.makeStringsEqual('100011000', '110111000') == True
assert solution.makeStringsEqual('1100', '0000') == False
assert solution.makeStringsEqual('10001', '01000') == True
assert solution.makeStringsEqual('00100011', '00101110') == True
assert solution.makeStringsEqual('101000', '000101') == True
assert solution.makeStringsEqual('10', '10') == True
assert solution.makeStringsEqual('01011', '10100') == True
assert solution.makeStringsEqual('1100111', '1010011') == True
assert solution.makeStringsEqual('111', '000') == False
assert solution.makeStringsEqual('11', '00') == False
assert solution.makeStringsEqual('0101010', '1101011') == True
assert solution.makeStringsEqual('0000', '1010') == False
assert solution.makeStringsEqual('00', '01') == False
assert solution.makeStringsEqual('10', '01') == True
assert solution.makeStringsEqual('11', '00') == False
assert solution.makeStringsEqual('0110011', '1010000') == True
assert solution.makeStringsEqual('11110001', '11010000') == True
assert solution.makeStringsEqual('00000110', '10110010') == True
assert solution.makeStringsEqual('000111', '001101') == True
assert solution.makeStringsEqual('10101', '01001') == True
assert solution.makeStringsEqual('0011111', '0001110') == True
assert solution.makeStringsEqual('1110011', '1011111') == True
assert solution.makeStringsEqual('11111101', '10100000') == True
assert solution.makeStringsEqual('000110000', '010001000') == True
assert solution.makeStringsEqual('00111', '01111') == True
assert solution.makeStringsEqual('10', '10') == True
assert solution.makeStringsEqual('101000', '101001') == True
assert solution.makeStringsEqual('11110100', '10111001') == True
assert solution.makeStringsEqual('100101', '101010') == True
assert solution.makeStringsEqual('10000', '01010') == True
assert solution.makeStringsEqual('001000101', '111000011') == True
assert solution.makeStringsEqual('0110000101', '1011011001') == True
assert solution.makeStringsEqual('000101', '000001') == True
assert solution.makeStringsEqual('11011', '00110') == True
assert solution.makeStringsEqual('0001011', '1101001') == True
assert solution.makeStringsEqual('1101', '1100') == True
assert solution.makeStringsEqual('11', '01') == True
assert solution.makeStringsEqual('110010110', '111000011') == True
assert solution.makeStringsEqual('00101001', '00000100') == True
assert solution.makeStringsEqual('1010111', '0001101') == True
assert solution.makeStringsEqual('100110011', '001111101') == True
assert solution.makeStringsEqual('11011', '10001') == True
assert solution.makeStringsEqual('01010', '11001') == True
assert solution.makeStringsEqual('1100110110', '0101101101') == True