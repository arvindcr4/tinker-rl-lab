
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
    def numDistinct(self, s: str, t: str) -> int:
        n = len(t)
        f = [1] + [0] * n
        for a in s:
            for j in range(n, 0, -1):
                if a == t[j - 1]:
                    f[j] += f[j - 1]
        return f[n]

solution=Solution()
assert solution.numDistinct('uk', 'rhu') == 0
assert solution.numDistinct('qgqmkyzz', 'qiejskhs') == 0
assert solution.numDistinct('hytnx', 'nz') == 0
assert solution.numDistinct('slmbty', 'rurrpz') == 0
assert solution.numDistinct('hukwjtmrt', 'ppkfj') == 0
assert solution.numDistinct('dkwqsiowosa', 'x') == 0
assert solution.numDistinct('jauglhjqq', 'qmzzu') == 0
assert solution.numDistinct('efui', 'm') == 0
assert solution.numDistinct('wrmk', 'szgg') == 0
assert solution.numDistinct('wu', 'yyfbgvx') == 0
assert solution.numDistinct('fz', 'ay') == 0
assert solution.numDistinct('wtuoy', 'gmmvxxeh') == 0
assert solution.numDistinct('tmrewqax', 'w') == 1
assert solution.numDistinct('cve', 'svp') == 0
assert solution.numDistinct('p', 'lnhgrdu') == 0
assert solution.numDistinct('wga', 'xymk') == 0
assert solution.numDistinct('wtxiuwh', 'ds') == 0
assert solution.numDistinct('rbcmas', 'x') == 0
assert solution.numDistinct('mavwj', 'shdqor') == 0
assert solution.numDistinct('hghzmqnq', 'wiookuz') == 0
assert solution.numDistinct('segmir', 'hozs') == 0
assert solution.numDistinct('dcxh', 'tzjsbor') == 0
assert solution.numDistinct('iwrmrjb', 'uyck') == 0
assert solution.numDistinct('hlpyuass', 'rrqm') == 0
assert solution.numDistinct('ywt', 'y') == 1
assert solution.numDistinct('ieqwy', 'ennpjh') == 0
assert solution.numDistinct('krxntbdywg', 'ddridarjm') == 0
assert solution.numDistinct('nxmvufrnbyl', 'lrgygdry') == 0
assert solution.numDistinct('ficsbascq', 'qkhgsn') == 0
assert solution.numDistinct('eaopbquhckj', 'ushk') == 0
assert solution.numDistinct('brno', 'fizoft') == 0
assert solution.numDistinct('vzltbq', 'cre') == 0
assert solution.numDistinct('aiqt', 'znoxiltt') == 0
assert solution.numDistinct('iar', 'fao') == 0
assert solution.numDistinct('xueihbvbm', 'mcne') == 0
assert solution.numDistinct('shciaxxynu', 'gzfx') == 0
assert solution.numDistinct('cnheq', 'ojthtlpxu') == 0
assert solution.numDistinct('z', 'wqesm') == 0
assert solution.numDistinct('qu', 'vyfdugja') == 0
assert solution.numDistinct('ekalijijo', 'mkzbyn') == 0
assert solution.numDistinct('mweesfk', 'cxjgbizth') == 0
assert solution.numDistinct('mxkvsgggt', 'nyhdfk') == 0
assert solution.numDistinct('rzkimgugyml', 'le') == 0
assert solution.numDistinct('fckklfde', 'ujlqzzzbq') == 0
assert solution.numDistinct('tzvyv', 'amwve') == 0
assert solution.numDistinct('ceh', 'icvhuzegcq') == 0
assert solution.numDistinct('sketav', 'ssbxhap') == 0
assert solution.numDistinct('wkvz', 'anqquoy') == 0
assert solution.numDistinct('iso', 'rpdrvu') == 0
assert solution.numDistinct('gddpvyeef', 'gvtqtswc') == 0
assert solution.numDistinct('t', 'sqcxg') == 0
assert solution.numDistinct('oue', 'wdrdugr') == 0
assert solution.numDistinct('mcayqhz', 'cmxylqz') == 0
assert solution.numDistinct('qje', 'oxmoftglqv') == 0
assert solution.numDistinct('e', 'a') == 0
assert solution.numDistinct('i', 'dhbzlt') == 0
assert solution.numDistinct('cpmkgcsnw', 'tepmohcotsl') == 0
assert solution.numDistinct('otb', 'oulxvypktnt') == 0
assert solution.numDistinct('wiy', 'lnty') == 0
assert solution.numDistinct('ibggdlat', 'baz') == 0
assert solution.numDistinct('ddmpsicss', 'taa') == 0
assert solution.numDistinct('gulg', 'wtkzhdu') == 0
assert solution.numDistinct('qketxt', 'mdp') == 0
assert solution.numDistinct('td', 'que') == 0
assert solution.numDistinct('kud', 'ubmkhlxgrtc') == 0
assert solution.numDistinct('qyjr', 'szzihljiy') == 0
assert solution.numDistinct('ew', 'peycmq') == 0
assert solution.numDistinct('ggxpcxododc', 'ldbialcpxpj') == 0
assert solution.numDistinct('fxrdfzz', 'mb') == 0
assert solution.numDistinct('y', 'qozopipd') == 0
assert solution.numDistinct('zo', 'o') == 1
assert solution.numDistinct('imavtf', 'mion') == 0
assert solution.numDistinct('dhp', 'lkjbnhh') == 0
assert solution.numDistinct('vwnzlsdxaj', 'ezlpmrft') == 0
assert solution.numDistinct('trnfwv', 'wilduy') == 0
assert solution.numDistinct('eaifakvhoxq', 'iujmq') == 0
assert solution.numDistinct('ll', 'naecle') == 0
assert solution.numDistinct('xkojcqfalxr', 'nmw') == 0
assert solution.numDistinct('ezkro', 'c') == 0
assert solution.numDistinct('j', 'hx') == 0
assert solution.numDistinct('kwlqjpw', 'ojpdufjx') == 0
assert solution.numDistinct('kwnd', 'rznf') == 0
assert solution.numDistinct('gikxw', 'qrclsa') == 0
assert solution.numDistinct('bdfziqrak', 'cbw') == 0
assert solution.numDistinct('xc', 'bbupxxfxg') == 0
assert solution.numDistinct('p', 'uvlogmtnnkn') == 0
assert solution.numDistinct('nfddsefsl', 'ja') == 0
assert solution.numDistinct('cvhu', 'udtw') == 0
assert solution.numDistinct('vedolsqwydq', 'gwjeubvepkx') == 0
assert solution.numDistinct('tpyi', 'm') == 0
assert solution.numDistinct('wvyewe', 'nkzuvf') == 0
assert solution.numDistinct('ybioqvfaj', 'y') == 1
assert solution.numDistinct('edqezteczf', 'xz') == 0
assert solution.numDistinct('wglmxzc', 'zzsdjs') == 0
assert solution.numDistinct('oyhovsx', 'lgcjq') == 0
assert solution.numDistinct('ztvavjytjuj', 'w') == 0
assert solution.numDistinct('e', 'rgyponhklt') == 0
assert solution.numDistinct('rzxbehzhkrl', 'lc') == 0
assert solution.numDistinct('tss', 'aru') == 0
assert solution.numDistinct('ymqmynkvc', 'gcas') == 0