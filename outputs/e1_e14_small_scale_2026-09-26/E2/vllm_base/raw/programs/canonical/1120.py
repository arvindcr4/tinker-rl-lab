
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

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        def dfs(root):
            if root is None:
                return 0, 0
            ls, ln = dfs(root.left)
            rs, rn = dfs(root.right)
            s = root.val + ls + rs
            n = 1 + ln + rn
            nonlocal ans
            ans = max(ans, s / n)
            return s, n

        ans = 0
        dfs(root)
        return ans

solution=Solution()
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55010>) == 56789.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55190>) == 96225.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55090>) == 94970.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55290>) == 81796.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55110>) == 69574.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55290>) == 49469.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b550d0>) == 68819.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55290>) == 40525.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55110>) == 79669.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b550d0>) == 48984.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55450>) == 66708.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b54fd0>) == 53622.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55390>) == 67652.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55290>) == 59807.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55510>) == 83686.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b555d0>) == 73012.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55490>) == 69087.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55250>) == 48509.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55410>) == 393.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55110>) == 59707.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55150>) == 77516.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55690>) == 91078.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55790>) == 92716.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55990>) == 64550.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55890>) == 73880.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55950>) == 56829.6
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55b50>) == 52247.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b558d0>) == 75958.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55a90>) == 3752.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55d50>) == 78475.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55050>) == 39076.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55b10>) == 42748.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b554d0>) == 46123.333333333336
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55ad0>) == 91333.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55090>) == 75526.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55dd0>) == 82676.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55f90>) == 62224.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55e10>) == 83821.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55c10>) == 29447.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55e50>) == 62891.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55e90>) == 83913.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56190>) == 93113.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56110>) == 60125.333333333336
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b561d0>) == 71848.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56410>) == 59250.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56090>) == 65471.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56490>) == 53105.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b565d0>) == 96641.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b55ed0>) == 57035.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56690>) == 33851.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56510>) == 79411.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56450>) == 94204.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56650>) == 65345.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56350>) == 4418.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b567d0>) == 44018.375
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56990>) == 44408.8
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56890>) == 47015.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56a90>) == 28622.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b563d0>) == 68488.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56790>) == 65222.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56b10>) == 60600.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56710>) == 50733.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56a50>) == 47195.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b569d0>) == 90688.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56d50>) == 48508.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56f10>) == 30452.6
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56cd0>) == 96634.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56e10>) == 99886.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56d10>) == 69649.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57090>) == 74868.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56f90>) == 94937.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57010>) == 76849.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57110>) == 87125.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b564d0>) == 68995.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b570d0>) == 98406.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56b90>) == 90042.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56d10>) == 94947.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b568d0>) == 70100.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57110>) == 40819.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56b90>) == 89305.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57010>) == 86138.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b568d0>) == 90480.33333333333
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57250>) == 32645.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56fd0>) == 79523.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b571d0>) == 25299.5
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57210>) == 63712.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56d10>) == 99171.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b564d0>) == 71415.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57250>) == 66112.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b57010>) == 67069.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b568d0>) == 60036.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56c90>) == 58689.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b571d0>) == 19434.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56ed0>) == 29257.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b568d0>) == 70655.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56c90>) == 98641.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b572d0>) == 65314.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56ed0>) == 81200.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b56fd0>) == 54759.0
assert solution.maximumAverageSubtree(<__main__.TreeNode object at 0x7f82d6b564d0>) == 87560.0