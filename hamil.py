import numpy as np
from .exact_diag import *
import itertools as its


def SS_ij(bl,bonds,Js):
    H = 0
    for (i,j),J in zip(bonds,Js): 
        H += J*SiSj(bl,i,j)
    return H
    
def Sxxz_ij(bl,bonds,Jxs,Jzs): # spin xxz model
    H = 0
    for (i,j),Jx,Jz in zip(bonds,Jxs,Jzs):
        H += Jx*SpSm(bl,i,j)+Jz*SzSz(bl,i,j)
    return H

# B*Sz_i
def Sz_i(bl,sites,Bs):
    H = 0
    for i,B in zip(sites,Bs):
        H += B*Sz(bl,i)
    return H

# A*Sz_i*Sz_i
def SzSz_i(bl,sites,As):
    H = 0
    for i,A in zip(sites,As):
        H += A*SzSz(bl,i,i)
    return H

