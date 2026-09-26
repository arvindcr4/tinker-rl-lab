
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
    def sortFeatures(self, features: List[str], responses: List[str]) -> List[str]:
        # Create a set of features for quick lookup
        feature_set = set(features)
        
        # Count popularity: number of responses that contain the feature
        popularity = {feature: 0 for feature in features}
        
        for response in responses:
            # Split response into words and get unique words
            words = set(response.split())
            # For each feature, check if it's in the response
            for feature in features:
                if feature in words:
                    popularity[feature] += 1
        
        # Sort features by popularity (descending), then by original index (ascending)
        # We need to preserve original indices for tie-breaking
        indexed_features = [(feature, i) for i, feature in enumerate(features)]
        
        # Sort by popularity descending, then by original index ascending
        indexed_features.sort(key=lambda x: (-popularity[x[0]], x[1]))
        
        return [feature for feature, _ in indexed_features]

solution=Solution()
assert solution.sortFeatures(['i', 'd', 'p', 'j', 'z', 'f', 'f', 'c', 'r', 'f'], ['j', 'z f p d f p', 'i p i i f i p p j', 'r i j d p i r', 'j i j d', 'f f d i d', 'd f']) == ['d', 'i', 'j', 'f', 'f', 'f', 'p', 'z', 'r', 'c']
assert solution.sortFeatures(['j', 'o', 'h', 'm', 'e', 'o', 'g', 'a', 'm', 'i'], ['o h h m', 'm i i m m m i m o', 'g m i m m e g j e', 'a j m m j o i', 'j m j m j', 'o m i o']) == ['m', 'm', 'o', 'o', 'i', 'j', 'h', 'e', 'g', 'a']
assert solution.sortFeatures(['a', 'r', 'r', 'd', 'o', 'b', 'w'], ['r']) == ['r', 'r', 'a', 'd', 'o', 'b', 'w']
assert solution.sortFeatures(['g', 'j', 'd', 'a', 'b'], ['g j a j a g b j', 'b', 'g d j g d b', 'g b j a a g b', 'd g a j d d b j', 'j']) == ['j', 'b', 'g', 'a', 'd']
assert solution.sortFeatures(['y', 't', 'r', 'w', 'g', 'j', 'u', 'c', 'p'], ['t w j p j g c t r w', 'p', 'j', 'u t y w g', 'r c c r p j c g w']) == ['w', 'g', 'j', 'p', 't', 'r', 'c', 'y', 'u']
assert solution.sortFeatures(['k', 'a', 'b', 'v', 'x', 'b', 'e', 'l', 'q'], ['q a q k', 'b l q a k q a a l', 'v a k', 'a q b v v l', 'b']) == ['a', 'k', 'b', 'b', 'q', 'v', 'l', 'x', 'e']
assert solution.sortFeatures(['x', 'r', 'm', 'k', 'p', 'h', 'l', 'f', 'l'], ['l l p h h l k', 'f', 'l f r p h p p', 'k x l m x m', 'p p m r l m h', 'l']) == ['l', 'l', 'p', 'h', 'r', 'm', 'k', 'f', 'x']
assert solution.sortFeatures(['u'], ['u u u u', 'u u u', 'u u u u u u u u', 'u u u u u u u u u', 'u u u u u u u', 'u u u u u u u u', 'u u u u u', 'u u u u u u u u']) == ['u']
assert solution.sortFeatures(['i', 'm', 't', 'h', 'q', 'z', 'j'], ['q j m q i', 'm', 'j', 't i z t m j t i j', 'j t z z q q h h j q', 'm q q m i t']) == ['m', 'j', 'i', 't', 'q', 'z', 'h']
assert solution.sortFeatures(['o', 's', 'x', 'z', 'd'], ['x s z', 's x s d d x o s z', 'd o']) == ['o', 's', 'x', 'z', 'd']
assert solution.sortFeatures(['r', 'b', 'u', 's', 'n', 'c', 'r'], ['n s b c s r b', 'c u r r u u r b', 'r s r c s n r', 'c u r s r n c u']) == ['r', 'c', 'r', 's', 'n', 'b', 'u']
assert solution.sortFeatures(['r', 'm', 'h', 'h', 's', 'y'], ['m y h h', 'h m s']) == ['m', 'h', 'h', 's', 'y', 'r']
assert solution.sortFeatures(['s', 'e', 'l', 'l', 'd', 'd', 'm', 'l', 's', 't'], ['d s l l l m l l l', 'e d l e t']) == ['l', 'l', 'd', 'd', 'l', 's', 'e', 'm', 's', 't']
assert solution.sortFeatures(['w', 'z', 's', 'g', 'h', 'c'], ['z h s w g', 'z w w s w c s h g c']) == ['w', 'z', 's', 'g', 'h', 'c']
assert solution.sortFeatures(['i', 'a', 'z', 'r'], ['r a z r a r r i z', 'z a a', 'r a z z z i r i r r', 'i a r r z z z z i', 'r a r z a i i i r', 'z z z r z r a', 'z z a z r z i']) == ['a', 'z', 'r', 'i']
assert solution.sortFeatures(['v', 'r', 'w', 'z', 'w', 'e', 'g', 'z'], ['r z r e w g g w', 'z w', 'w z z z v w v z', 'w w v', 'r g e w', 'e', 'v', 'g e', 'r w g g e z', 'g w g w w z w']) == ['w', 'w', 'z', 'e', 'g', 'z', 'v', 'r']
assert solution.sortFeatures(['u', 'f', 'q', 'f', 't', 'g', 'w', 'a', 'p'], ['a t t a', 'f f q g a', 'g a a p w']) == ['a', 'g', 'f', 'q', 'f', 't', 'w', 'p', 'u']
assert solution.sortFeatures(['r', 'z', 'j', 'u', 'i', 'm', 'm', 'k', 'a'], ['a j r a i m j i m m', 'k r m k', 'i m', 'u', 'u r k', 'z m a a r i', 'm r', 'r u i k z k u', 'a r m m r r z r', 'j a j j k i r m']) == ['r', 'm', 'm', 'i', 'k', 'a', 'z', 'u', 'j']
assert solution.sortFeatures(['c', 'y', 'e'], ['y e e c e e e c c c']) == ['c', 'y', 'e']
assert solution.sortFeatures(['f', 't', 'e', 's', 'h', 'j', 'y', 'p', 's', 'q'], ['y s h', 'e y s e h e j s s', 'y f q f t', 'p s', 'y e j f s h s q f', 's j f', 'q t y h', 's s f s j', 'f t', 'j t']) == ['s', 's', 'f', 'j', 'y', 't', 'h', 'q', 'e', 'p']
assert solution.sortFeatures(['i', 't', 'b', 'u', 't'], ['t i i i b t u', 'u b i t t i', 't t t b u i t', 'u t t b b b', 'b t b i i u t i', 'b b b t', 't i t', 't', 'b u u t i u t t']) == ['t', 't', 'b', 'i', 'u']
assert solution.sortFeatures(['f', 'g', 'p', 'k', 'n'], ['n g k n g n f g g', 'n p k g k f f k k', 'p p k n f f n p g n', 'n k k k k f', 'n p k', 'p k f f k n', 'n n k n n', 'n g p']) == ['n', 'k', 'f', 'p', 'g']
assert solution.sortFeatures(['a', 'o'], ['o o', 'a', 'o o o o', 'o a o a', 'a o o a a a', 'o o a o o o o', 'o a']) == ['o', 'a']
assert solution.sortFeatures(['g', 'f', 'k', 'w', 's'], ['f w k k k g', 'f f f s g', 'f g g', 's', 'w', 'k k k g s s k', 'g s k w f g k k', 'w k k f s s k', 'f w f k g', 'g k g w s g']) == ['g', 'f', 'k', 'w', 's']
assert solution.sortFeatures(['m', 'k', 'v', 'v', 'r'], ['r m m v v', 'm r m v r', 'v r v k v m m', 'v v v m k m v', 'm v', 'k m r', 'v v v m m v v m r', 'r r k', 'm v m v k v']) == ['m', 'v', 'v', 'r', 'k']
assert solution.sortFeatures(['r', 'u', 'e', 'x'], ['r x e r e e x u r', 'u r e e', 'r u r u']) == ['r', 'u', 'e', 'x']
assert solution.sortFeatures(['v', 't', 'u', 'v', 'n', 'v'], ['u u n u v u v u t', 'v n', 'v v', 'u t t v n u u n u', 'n u v v v']) == ['v', 'v', 'v', 'n', 'u', 't']
assert solution.sortFeatures(['v', 'c', 'v', 'x', 'j', 'a', 'a'], ['a x c j v a a a', 'c x a v x a', 'c j', 'x v v c j a c a v', 'a v v x', 'j', 'a v v x x', 'x c j j c c v j a v', 'x c a a a a c a a v', 'x x a j v a j a v v']) == ['v', 'v', 'x', 'a', 'a', 'c', 'j']
assert solution.sortFeatures(['s', 'i'], ['i i s', 'i s i i s s i s', 'i i i i s s', 's i i s i s s s', 'i', 's s i s']) == ['i', 's']
assert solution.sortFeatures(['u', 'h', 'a', 'u', 'r', 'd', 'n', 's', 'v', 's'], ['s r n h s n', 'h u a s v s d r d', 'v s r', 'a a d s h u d d u r', 's d', 'r r v a', 'd s']) == ['s', 's', 'r', 'd', 'h', 'a', 'v', 'u', 'u', 'n']
assert solution.sortFeatures(['p', 'e', 'c', 'a', 'm', 'h'], ['m m m c', 'c e m']) == ['c', 'm', 'e', 'p', 'a', 'h']
assert solution.sortFeatures(['y', 'e'], ['e e y e', 'e y y y y y y y e', 'e e', 'y y y y y e e e e y', 'y e y e y e', 'e y e y', 'e e e y e']) == ['e', 'y']
assert solution.sortFeatures(['s', 'n', 'q', 'k', 'j', 'a', 'k', 'j', 'u', 'v'], ['q q', 'n', 'u n j j', 'k k s j u a', 'k u', 'a a u j n n a k k s', 'v a q', 'j u j a q j', 'n k j j j k v k']) == ['j', 'j', 'u', 'n', 'k', 'a', 'k', 'q', 's', 'v']
assert solution.sortFeatures(['o', 'r', 'e', 'x', 'q', 'w'], ['q q', 'q r q q r w x q e x']) == ['q', 'r', 'e', 'x', 'w', 'o']
assert solution.sortFeatures(['s'], ['s s s s s s', 's s s s s s s s', 's s s s s s', 's s s s s s s s', 's s s s s s s s s', 's s s s', 's s s s s s s s', 's']) == ['s']
assert solution.sortFeatures(['q', 'n', 'q', 't', 'r', 'p', 'r', 'd'], ['d', 'n q d d p p r r p', 'd r q d p q q p', 'r d t', 'r n p t r r n p p', 'r r p r r d', 't q q', 'q q t n r', 'p q d q q']) == ['r', 'r', 'd', 'q', 'q', 'p', 't', 'n']
assert solution.sortFeatures(['f', 'w', 'u', 'z', 'b', 'y', 'g', 'e'], ['z b e u y e f b g']) == ['f', 'u', 'z', 'b', 'y', 'g', 'e', 'w']
assert solution.sortFeatures(['i'], ['i i i i i i i i i', 'i i i i i i i i i']) == ['i']
assert solution.sortFeatures(['u', 'n', 'h', 'p', 'e', 'l', 'e', 'i', 'a'], ['p l u h', 'u u i', 'n e n a a a h', 'i u', 'i h n l u u e e i u']) == ['u', 'h', 'i', 'n', 'e', 'l', 'e', 'p', 'a']
assert solution.sortFeatures(['s', 'q'], ['q q q s q q', 'q', 's s q q q s', 's q s', 's s q s q q s s q s']) == ['q', 's']
assert solution.sortFeatures(['c', 'j'], ['c', 'c j', 'j j c j j c j c', 'c', 'j j c c j j', 'c c j j c c j', 'j j', 'c', 'j c j j j']) == ['c', 'j']
assert solution.sortFeatures(['n', 'o', 'e'], ['e n e o n', 'n n n n n o n o o', 'e o o o o o n o o o', 'e n e n e o e', 'n n n o e o n', 'o n e n o e e o e', 'e o o n n n o', 'o e n o']) == ['n', 'o', 'e']
assert solution.sortFeatures(['j', 'l', 'z', 'v'], ['l', 'j z l j j l v', 'l v', 'j j v l j l v l l j']) == ['l', 'v', 'j', 'z']
assert solution.sortFeatures(['y', 'x', 'q', 'm', 'u', 'm', 'i', 'v', 'q', 'd'], ['v d m', 'm']) == ['m', 'm', 'v', 'd', 'y', 'x', 'q', 'u', 'i', 'q']
assert solution.sortFeatures(['o', 'r', 'g', 's', 'b', 'j'], ['s s g r g r b o s', 'r o b o o g r', 'g r j s b r', 'o s o s', 's r j j b', 'r s s o', 's s o j s b b b']) == ['s', 'o', 'r', 'b', 'g', 'j']
assert solution.sortFeatures(['n', 'n', 'i', 'c'], ['n n i n n c', 'n n n', 'c c n i', 'i n i n c i n c n', 'n n c n n c c n n']) == ['n', 'n', 'c', 'i']
assert solution.sortFeatures(['x', 't', 'q', 'r', 'w', 'z', 'x', 'y', 's'], ['q']) == ['q', 'x', 't', 'r', 'w', 'z', 'x', 'y', 's']
assert solution.sortFeatures(['s', 'p', 'l', 'e', 'g', 'k', 'm', 'i', 'e', 'z'], ['s p g e', 'k m e', 'p g p g s l z s', 'm m m l', 'l g g e z e i p', 'e z p l g p s i', 'm', 'l p g l i']) == ['p', 'l', 'g', 'e', 'e', 's', 'm', 'i', 'z', 'k']
assert solution.sortFeatures(['y', 'f', 'k', 'u'], ['y f u u', 'f u u y u f k y', 'u f u f', 'u k k u', 'y y y f f k k y y', 'k k u', 'u u y u f u f y k f', 'f y k y f f y f y u']) == ['u', 'f', 'k', 'y']
assert solution.sortFeatures(['q', 'b', 'h', 'z', 'm', 'd'], ['h d d h m h z', 'z d b q q m q q b m']) == ['z', 'm', 'd', 'q', 'b', 'h']
assert solution.sortFeatures(['c', 'q', 'i', 'v', 'i', 'w', 'k'], ['i w w i', 'w c c i v q i k i', 'v w i k', 'k c k c k q c w i w']) == ['i', 'i', 'w', 'k', 'c', 'q', 'v']
assert solution.sortFeatures(['m', 'l'], ['l', 'm l m m l l l l m', 'l m l l l l l m', 'l l m m m', 'm m m m m l l l', 'l l l l m l', 'm l l l l', 'l']) == ['l', 'm']
assert solution.sortFeatures(['z', 'g', 'g', 'a', 'j', 'd', 'j', 'o', 'x'], ['x', 'j j a d d x j', 'j g', 'g']) == ['g', 'g', 'j', 'j', 'x', 'a', 'd', 'z', 'o']
assert solution.sortFeatures(['p', 'r', 't'], ['t t', 'p r r r p p', 't p p p p p p p', 'r', 'r p p r r p t p', 't']) == ['t', 'p', 'r']
assert solution.sortFeatures(['w', 's', 'c', 'k', 'z', 'z', 'q'], ['z w w s w z c c s z', 'c z w s', 'q z k w', 'z c', 'k w k', 's z s s z', 'z c z s z w q', 'q w z z w z k k s k']) == ['z', 'z', 'w', 's', 'c', 'k', 'q']
assert solution.sortFeatures(['l', 'j'], ['l j l l j j j', 'l l j j j j l l', 'j j j', 'j j j l j l', 'l', 'l', 'j j l j l l l j j j', 'l', 'l l j l l l l l j j']) == ['l', 'j']
assert solution.sortFeatures(['f', 'd', 'j', 'v', 't', 'k', 'l', 'k', 'v'], ['k k v l k l v', 'l', 'k k t t k k k l v k', 'v k t', 't d k v', 'f f j v l f l v', 'd k t v k']) == ['v', 'v', 'k', 'k', 't', 'l', 'd', 'f', 'j']
assert solution.sortFeatures(['d', 'c', 'd', 'p', 'p', 'a', 'f', 'd'], ['d d f p d p d p', 'f c p d', 'd f p p f a p c d']) == ['d', 'd', 'p', 'p', 'f', 'd', 'c', 'a']
assert solution.sortFeatures(['p', 'p', 'j'], ['p p j j p p p j p p', 'j p p p p j', 'p j p', 'p j j p p j p p p p', 'p', 'j j p p j p j p p', 'p j j p j', 'p', 'p p j p j j p', 'p p p p j p']) == ['p', 'p', 'j']
assert solution.sortFeatures(['s', 'c', 'i', 'v', 'd', 'e', 'g', 'o', 'y'], ['d g d o o v', 'd', 'i d y o v d y v o']) == ['d', 'v', 'o', 'i', 'g', 'y', 's', 'c', 'e']
assert solution.sortFeatures(['n', 'o', 'd', 'g', 'l', 'h', 'u', 'x', 'p', 's'], ['x p g p', 'p n g d u n p o', 'n h', 'l h d', 'h o p l u s h l p', 's p g o h', 'p l o p s x p p g']) == ['p', 'o', 'g', 'h', 'l', 's', 'n', 'd', 'u', 'x']
assert solution.sortFeatures(['e', 'v', 'i', 'k', 'd', 'p', 'a'], ['p v v k i a p v e', 'k p e', 'i k k', 'i a k a a e e i', 'p k p k p', 'v i a i a', 'k p d d v e p i v v', 'e d p', 'a e i p p e']) == ['e', 'i', 'k', 'p', 'a', 'v', 'd']
assert solution.sortFeatures(['k', 's', 'v', 'g', 'e', 'l'], ['v l l e l k e v v s', 's s', 's e g', 'l l g l k e v g g', 'v e s s']) == ['s', 'e', 'v', 'k', 'g', 'l']
assert solution.sortFeatures(['j', 's', 'o'], ['o o', 'o', 's j o s', 'o o o j j', 's o s j s s j j', 'j j s', 's s s o o s o o j o', 'o o j j j']) == ['o', 'j', 's']
assert solution.sortFeatures(['a', 'q', 'o', 'o', 'l', 'w', 'n', 'a', 'n', 'v'], ['l l l v a n o', 'o n n l a v n', 'v v a o n o a n w', 'a n o o a o n']) == ['a', 'o', 'o', 'n', 'a', 'n', 'v', 'l', 'w', 'q']
assert solution.sortFeatures(['n', 'f', 'p', 'd', 'p', 'c', 'a', 'r', 'e'], ['n a p e c n r f', 'n a n d n f a', 'd p p d', 'p f f p n d r', 'e a p a a r', 'p n n n f c a n e', 'a p c p f r d r p', 'p r n f r p f a r p', 'p p p f d p e d', 'n p f e d p r p']) == ['p', 'p', 'f', 'n', 'd', 'a', 'r', 'e', 'c']
assert solution.sortFeatures(['w', 't', 'e', 'b', 'q', 'a', 'k', 'z', 'h', 'w'], ['w w', 'a q k e z w w h', 'q h', 'q z w b h w w h e e', 'a a a a h z k w t e', 'w']) == ['w', 'w', 'h', 'e', 'q', 'z', 'a', 'k', 't', 'b']
assert solution.sortFeatures(['s', 'h', 'o', 'j', 'k', 'h', 'g', 'b', 'e'], ['e h k', 'b h b h e g g g k', 'h g', 'h g s j k', 'e']) == ['h', 'h', 'k', 'g', 'e', 's', 'j', 'b', 'o']
assert solution.sortFeatures(['y', 'r', 's', 'n', 'n', 'p', 'u'], ['r', 'r u n u p', 'u r p n r n n', 'r n n n n s n', 'r n p']) == ['r', 'n', 'n', 'p', 'u', 's', 'y']
assert solution.sortFeatures(['j', 'r', 'm', 'n'], ['n j m j n n', 'm r j m r n r j m', 'j j j m n n', 'r n n r r', 'm', 'j n r j m']) == ['m', 'n', 'j', 'r']
assert solution.sortFeatures(['k', 'f', 'k', 'y', 'w', 'u', 'z'], ['f w y']) == ['f', 'y', 'w', 'k', 'k', 'u', 'z']
assert solution.sortFeatures(['h', 'n', 'p', 'b', 'h', 'k', 'o', 't'], ['t n o h h h p n', 'p k h b p b n', 'p t t k h', 'o k p t', 'k', 'k o h', 'n t t b h h k h', 'b h p k', 'h p', 'b b h h h o h b h b']) == ['h', 'h', 'k', 'p', 'b', 'o', 't', 'n']
assert solution.sortFeatures(['i', 'o', 'v', 'f', 'k', 'e', 'y', 'l', 'e'], ['k i y y f i', 'i y l f e', 'y i l k y v y e f', 'k y y v y f e', 'l i y e i o l', 'k l l k o l o y', 'i y', 'v f o i o e e e e', 'e l v o k f f']) == ['y', 'i', 'f', 'e', 'e', 'k', 'l', 'o', 'v']
assert solution.sortFeatures(['y', 'q', 'v', 'g', 'x', 'b', 'o', 'q', 'q'], ['q g o q q q', 'x', 'q q g q q x y g q q', 'x q v q x y b q', 'q v b y g q o', 'o b o y q b q y v', 'o v y y b q g v v']) == ['q', 'q', 'q', 'y', 'v', 'g', 'b', 'o', 'x']
assert solution.sortFeatures(['i', 'v', 'j', 'l', 'd', 'c'], ['j d', 'v v j j j i c d c', 'i d c j l v c i i v', 'd i l i i d v j', 'c v v i l v i c v c']) == ['i', 'v', 'j', 'd', 'l', 'c']
assert solution.sortFeatures(['y', 'c', 't', 'm', 'l', 'k', 'r'], ['l l', 'm t', 'c t r c r k t', 'l c t l y', 'm m', 'k t y l r y y', 't c l c y r t', 't c m y y m c t y r', 'l y y r k l t k', 'y c c']) == ['t', 'y', 'c', 'l', 'r', 'm', 'k']
assert solution.sortFeatures(['x', 'd', 'a', 'x', 'p', 'n', 'd', 'd'], ['a a a x a n d n d', 'x d d d a d a', 'd x p', 'a n a d d x d p d n', 'd d p n x p d x', 'x x d a', 'd d x d d a a n x', 'n x']) == ['x', 'x', 'd', 'd', 'd', 'a', 'n', 'p']
assert solution.sortFeatures(['f', 'h', 'c', 't', 'b', 'y', 'l', 'g', 'h'], ['l']) == ['l', 'f', 'h', 'c', 't', 'b', 'y', 'g', 'h']
assert solution.sortFeatures(['o', 'q', 'r', 'b', 'p', 't'], ['p r o p o b p', 't b o t q q r o q q', 'r o b o q p b']) == ['o', 'r', 'b', 'q', 'p', 't']
assert solution.sortFeatures(['k', 'e', 'g'], ['e e k k k e g', 'e e k g e e e k e g', 'e g g e g g k k k e', 'k g', 'k g e e e e k e', 'g e k']) == ['k', 'g', 'e']
assert solution.sortFeatures(['j', 'a', 'l', 'j', 'e', 'x', 'l', 'o', 'g'], ['a o g g l x g', 'x e a e j e o x', 'j x j g l l e g']) == ['x', 'j', 'a', 'l', 'j', 'e', 'l', 'o', 'g']
assert solution.sortFeatures(['k', 'f', 'v', 'x', 'u', 'w'], ['x k v v v', 'k u w x w k x v u', 'f v k u u v x u f w', 'w w', 'w f k w u x k k x', 'f w k w v u w v w', 'f k u f f f f v x x', 'v k f', 'x u u f f x', 'u x k k w v v f x u']) == ['k', 'f', 'v', 'x', 'u', 'w']
assert solution.sortFeatures(['j', 'a'], ['j j j j', 'a a j j', 'a j a']) == ['j', 'a']
assert solution.sortFeatures(['r', 'u'], ['r u r u r r u r u u', 'u r r u', 'u u r r u r r u', 'r u u u u r r u r']) == ['r', 'u']
assert solution.sortFeatures(['s', 'p', 'l', 'h', 'o'], ['l l h l s p s h o o', 'o o o l h o s', 's', 's h p h', 'o', 'l o o', 'o', 'p o o']) == ['o', 's', 'p', 'l', 'h']
assert solution.sortFeatures(['q', 'b', 'w'], ['q q w w', 'b w w q q w b b', 'b w', 'w w q q q w w q', 'q q w q q q q b q b', 'w b b b b b', 'q q b q w q b b', 'w w w', 'q q', 'b']) == ['w', 'q', 'b']
assert solution.sortFeatures(['k', 'h', 'a'], ['k a h a a h a a', 'k k k a a a k k a h', 'h', 'a k k h h k a a h', 'k h a a k a h h k', 'k h']) == ['h', 'k', 'a']
assert solution.sortFeatures(['j', 'r', 'l', 'm', 'n', 'v', 'h', 'f'], ['r l h h m h r l v']) == ['r', 'l', 'm', 'v', 'h', 'j', 'n', 'f']
assert solution.sortFeatures(['v', 's', 'h', 't'], ['s', 'h t', 'h s h s h h s', 'v s s s', 't v t t s v h h h v', 'v t s s t']) == ['s', 'v', 'h', 't']
assert solution.sortFeatures(['g', 'a', 'z', 'c', 'z', 'u', 'b'], ['c z', 'z b', 'a u c c', 'b z z a', 'u a z u a a a a a b']) == ['z', 'z', 'a', 'b', 'c', 'u', 'g']
assert solution.sortFeatures(['w', 'o'], ['w o w w w o w', 'o o o o w o o', 'w o', 'w o w w o w o o']) == ['w', 'o']
assert solution.sortFeatures(['k', 'e', 'u', 'z', 'k', 'a', 'l', 'f', 'b'], ['l z b', 'k a k u a e l', 'k z a z k k k', 'l k k', 'b l k k', 'e k', 'z k l u z a f f k k', 'f f f a', 'z z u e z a b', 'b a z e k k l u u l']) == ['k', 'k', 'a', 'l', 'z', 'e', 'u', 'b', 'f']
assert solution.sortFeatures(['e', 'k', 'y', 'n', 'p'], ['n k k n e p y', 'y e', 'e p p y k n', 'y n e e p k p n k', 'e e n p n k n y', 'e y k', 'y y k n n p', 'y n n k n']) == ['y', 'k', 'e', 'n', 'p']
assert solution.sortFeatures(['k', 'i', 'k', 'n', 'w', 'n', 'r', 's', 'h', 'y'], ['n i', 'n y y n', 's k i i y y w', 'n', 's n s k', 'w i', 'i k k n n h n h i', 'y s k r n w y', 'n h y k r n h i']) == ['n', 'n', 'k', 'i', 'k', 'y', 'w', 's', 'r', 'h']
assert solution.sortFeatures(['b', 'f', 'w', 'o', 'b', 'c', 'c', 'f', 'g'], ['c', 'c g c b g g g b c f', 'b']) == ['b', 'b', 'c', 'c', 'f', 'f', 'g', 'w', 'o']
assert solution.sortFeatures(['p'], ['p p', 'p', 'p p p p p p p p p', 'p p p', 'p p p p p p p p p p', 'p', 'p', 'p p p', 'p p p p p p', 'p']) == ['p']
assert solution.sortFeatures(['x', 'w', 'w', 's', 'q', 'v'], ['q s q s s']) == ['s', 'q', 'x', 'w', 'w', 'v']
assert solution.sortFeatures(['b', 'y', 'h', 'm'], ['h b m b h h b h y h', 'b y b b m', 'm y m y', 'b h b b']) == ['b', 'y', 'm', 'h']
assert solution.sortFeatures(['e', 'm', 't', 'd'], ['e e m e t d m', 'e m t t d e t', 't t t e t', 'd e d t d d d t m', 't d m e', 't t e e m t d e m m']) == ['e', 't', 'm', 'd']
assert solution.sortFeatures(['m', 'm', 'a', 'j'], ['j', 'a m a j m j']) == ['j', 'm', 'm', 'a']