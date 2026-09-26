
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
    def minimumCost(self, sentence: str, k: int) -> int:
        words = sentence.split()
        n = len(words)
        
        # dp[i] = minimum cost to arrange words[i:] into rows
        # We want dp[0]
        # Base case: dp[n] = 0 (no words left, no cost)
        
        # For each starting index i, we try to put words[i..j] in the first row
        # The length of the row would be sum of lengths of words[i..j] + (j - i) spaces
        # If this length <= k, then we can form a row, and the cost for this row is (k - length)^2
        # But note: if this is the last row (i.e., j == n-1), then we don't add the cost for this row.
        
        # dp[i] = min over all valid j >= i where words[i..j] fit in one row:
        #   if j == n-1: dp[j+1]  (no cost added for last row)
        #   else: (k - length)^2 + dp[j+1]
        
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        for i in range(n - 1, -1, -1):
            # Try all possible end positions j for the first row starting at i
            current_length = 0
            for j in range(i, n):
                # Add word[j] to the current row
                if j > i:
                    current_length += 1  # for the space
                current_length += len(words[j])
                
                if current_length > k:
                    break
                
                # Valid row from i to j
                if j == n - 1:
                    # Last row, no cost
                    cost = dp[j + 1]
                else:
                    cost = (k - current_length) ** 2 + dp[j + 1]
                
                if cost < dp[i]:
                    dp[i] = cost
        
        return dp[0]

solution=Solution()
assert solution.minimumCost('u s a m d i g v', 1) == 0
assert solution.minimumCost('r u p z u l', 8) == 1
assert solution.minimumCost('h z a', 7) == 0
assert solution.minimumCost('j b m d', 2) == 3
assert solution.minimumCost('p f w g z x s b', 1) == 0
assert solution.minimumCost('h p e r x b b m b j', 2) == 9
assert solution.minimumCost('k m w h v b x', 2) == 6
assert solution.minimumCost('h k a k', 6) == 1
assert solution.minimumCost('z s f q x q e', 9) == 0
assert solution.minimumCost('k e v t t z l l v', 8) == 2
assert solution.minimumCost('c j', 7) == 0
assert solution.minimumCost('y x n b c n v e x t', 4) == 4
assert solution.minimumCost('a', 6) == 0
assert solution.minimumCost('j t d z', 9) == 0
assert solution.minimumCost('i p g t p u x a c d', 10) == 1
assert solution.minimumCost('b i c v q c j t e b', 6) == 3
assert solution.minimumCost('x w h i p', 4) == 2
assert solution.minimumCost('r i x i o j l', 4) == 3
assert solution.minimumCost('l e j', 7) == 0
assert solution.minimumCost('n l m m t t d', 9) == 0
assert solution.minimumCost('e i b k y n y x y', 4) == 4
assert solution.minimumCost('w m y h a r d m', 7) == 0
assert solution.minimumCost('e s q l b y m f u', 5) == 0
assert solution.minimumCost('q n p y s q z e o', 4) == 4
assert solution.minimumCost('v f t o q f', 1) == 0
assert solution.minimumCost('l g y i w s t i', 4) == 3
assert solution.minimumCost('j v r', 3) == 0
assert solution.minimumCost('t j g n y', 6) == 1
assert solution.minimumCost('q d e x z o k z o n', 10) == 1
assert solution.minimumCost('i m', 1) == 0
assert solution.minimumCost('u p d h', 3) == 0
assert solution.minimumCost('y s x l l g i r p f', 8) == 2
assert solution.minimumCost('i', 8) == 0
assert solution.minimumCost('k t m t o', 4) == 2
assert solution.minimumCost('z d v s y n i', 2) == 6
assert solution.minimumCost('e b h q e u m j k g', 5) == 0
assert solution.minimumCost('k g m p c u d p j m', 9) == 0
assert solution.minimumCost('t c s w d', 1) == 0
assert solution.minimumCost('l m', 6) == 0
assert solution.minimumCost('z a f', 4) == 1
assert solution.minimumCost('z z j i y y v y', 10) == 1
assert solution.minimumCost('l', 9) == 0
assert solution.minimumCost('c k i y u y w', 2) == 6
assert solution.minimumCost('v p', 2) == 1
assert solution.minimumCost('u y u f l c a e y', 2) == 8
assert solution.minimumCost('w x f g y n', 7) == 0
assert solution.minimumCost('e u j c', 3) == 0
assert solution.minimumCost('v h e v', 5) == 0
assert solution.minimumCost('q l m m q n q b z u', 3) == 0
assert solution.minimumCost('j f v f p', 3) == 0
assert solution.minimumCost('h v c h v c l b n', 9) == 0
assert solution.minimumCost('d o v v', 7) == 0
assert solution.minimumCost('r f e w c', 7) == 0
assert solution.minimumCost('r', 6) == 0
assert solution.minimumCost('d z k r n q', 9) == 0
assert solution.minimumCost('o s', 10) == 0
assert solution.minimumCost('y s j j y s q j', 6) == 2
assert solution.minimumCost('i d w z j f', 4) == 2
assert solution.minimumCost('z v x l', 10) == 0
assert solution.minimumCost('d r', 1) == 0
assert solution.minimumCost('p l l q t l', 2) == 5
assert solution.minimumCost('f s d z n', 4) == 2
assert solution.minimumCost('d i n x', 9) == 0
assert solution.minimumCost('e v m c n w y j c h', 1) == 0
assert solution.minimumCost('p', 10) == 0
assert solution.minimumCost('o', 3) == 0
assert solution.minimumCost('m v z l o m', 8) == 1
assert solution.minimumCost('l', 2) == 0
assert solution.minimumCost('v j g p e d z e d c', 9) == 0
assert solution.minimumCost('g w o w', 3) == 0
assert solution.minimumCost('k n j w v a', 1) == 0
assert solution.minimumCost('y z', 3) == 0
assert solution.minimumCost('j g l p u z d g', 4) == 3
assert solution.minimumCost('e m b d b', 1) == 0
assert solution.minimumCost('s g e a f f q b c', 5) == 0
assert solution.minimumCost('f m b k v e v e i b', 1) == 0
assert solution.minimumCost('p b m h f b q', 2) == 6
assert solution.minimumCost('c m a o u z h s j', 6) == 2
assert solution.minimumCost('n g r', 7) == 0
assert solution.minimumCost('a p h z e w', 2) == 5
assert solution.minimumCost('o n w', 6) == 0
assert solution.minimumCost('m q v y t i d n x v', 4) == 4
assert solution.minimumCost('e s a m q r x', 5) == 0
assert solution.minimumCost('j e g t v l k', 10) == 1
assert solution.minimumCost('f l l f f k', 4) == 2
assert solution.minimumCost('c z g', 6) == 0
assert solution.minimumCost('l', 3) == 0
assert solution.minimumCost('k n l', 2) == 2
assert solution.minimumCost('x p k h j h z v', 3) == 0
assert solution.minimumCost('u v p s g', 7) == 0
assert solution.minimumCost('k j g h', 8) == 0
assert solution.minimumCost('j m j p t v', 3) == 0
assert solution.minimumCost('u e v u', 10) == 0
assert solution.minimumCost('p l r l t f', 4) == 2
assert solution.minimumCost('c p', 7) == 0
assert solution.minimumCost('l a i q u', 1) == 0
assert solution.minimumCost('s p a f h e b z s k', 9) == 0
assert solution.minimumCost('c x l d n f c', 10) == 1
assert solution.minimumCost('p', 7) == 0
assert solution.minimumCost('m p', 10) == 0