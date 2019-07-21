from Susan import *
import numpy as np

S = Susan("examples/tux/tux.png")
"""
i = 7
j = 10
S._nbd_check(i,j)
print(S._nbd_compare(i,j,5))
"""

#print(S.mask_nbd)

S.detect_edges(1)