from .assist import *
import numpy as np
import scipy.sparse as sp
import itertools as its

# only accept b>=0
def bittest(b,i):
    return b&(1<<i)
def bitflip(b,i):
    return b^(1<<i)
def bitcount(b,i=0):
    return (b>>i).bit_count()
def commute(b,i=0):
    return 1-2*(bitcount(b,i+1)%2)

def bitsflip(b,M): # flip all bits
    for i in range(M): b = bitflip(b,i)
    return b


# we want 0 to be also regarded as positive
# def sign(b):
    # return 2*(b>=0)-1
def sign(b):
    return 1



# def cd(b,i):  # c_dn
#     if b is None or not bittest(abs(b), i): return None
#     return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)
# def cu(b,i):  # c_up
#     if b is None or bittest(abs(b), i): return None
#     return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)
# def cd(b,i):  # c_dn
#     if b is None or not bittest(abs(b), i): return None
#     return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)
# def cu(b,i):  # c_up
#     if b is None or bittest(abs(b), i): return None
#     return commute(abs(b),i)*sign(b)*bitflip(abs(b),i)


# building block for S operators
def cucd(b,i):  # cdag_up c_dn
    if not bittest(abs(b),i): return None
    return sign(b)*bitflip(abs(b),i)
def cdcu(b,i):  # cdag_dn c_up
    if bittest(abs(b),i): return None
    return sign(b)*bitflip(abs(b),i)
def nd(b,i):
    if not bittest(abs(b),i): return None
    return b
def nu(b,i):
    if bittest(abs(b),i): return None
    return b

# we only swap between 01 and 10, otherwise return None
def bitswap(b,i,j):
    if bool(bittest(b,i))^bool(bittest(b,j)): return sign(b)*bitflip(bitflip(abs(b),j),i)
    return None

def opts(b,opt,idx):
    for i,o in zip(reversed(idx),reversed(opt)): b = o(b,i)
    return b



# 0 up 1 dn
def build_basis(M,Nu):
    basis = []; Nd = M-Nu
    minr = 0; maxr = 0
    for i in range(Nd):
        minr += 2**i
        maxr += 2**(M-i-1)
    for i in range(minr,maxr+1):
        nbit = 0
        for j in range(M):
            if bittest(i,j): nbit += 1
        if nbit==Nd:
            basis.append(i)
    return basis

# under descending order of Nu: M,M-1,...,0
def build_full_basis(M):
    Bl = []
    for Nd in range(M+1):
        bl = build_basis(M, M-Nd)
        Bl.append(bl)
    return Bl

def map2mtx(bi,bf):
    row,col,val = [],[],[]
    for n,(i,f) in enumerate(zip(bi,bf)):
        if i!=None and f!=None:
            col.append(n)
            row.append(bi.index(abs(f)))
            val.append(sign(f)/sign(i))
    return sp.csr_array((val,(row,col)), shape=(len(bf),len(bi)))


def Sx(bl,i):
    m = 1/2*map2mtx(bl, [cucd(b,i) for b in bl])   \
      + 1/2*map2mtx(bl, [cdcu(b,i) for b in bl])
    return m  
def Sy(bl,i):
    m = -1j/2*map2mtx(bl, [cucd(b,i) for b in bl]) \
      +  1j/2*map2mtx(bl, [cdcu(b,i) for b in bl])
    return m

def szi(b,i): return -bool(bittest(abs(b),i))+0.5
def Sz(bl,i):
    # return (Nu(bl,i)-Nd(bl,i))/2
    return sp.diags([szi(b,i) for b in bl])

def Sp(bl,i): return map2mtx(bl, [cucd(b,i) for b in bl])

def Sm(bl,i): return map2mtx(bl, [cdcu(b,i) for b in bl])

def Nu(bl,i): return map2mtx(bl, [nu(b,i) for b in bl])

def Nd(bl,i): return map2mtx(bl, [nd(b,i) for b in bl])

# def SiSj(bl,i,j):
    # m = Sp(bl,i)@Sm(bl,j)/2 +Sm(bl,i)@Sp(bl,j)/2 +Sz(bl,i)@Sz(bl,j)
    # return m
    
def SpSm(bl,i,j): # inculding h.c.
    Nb = len(bl)
    if i==j: return sp.eye(Nb)/2
    return map2mtx(bl, [bitswap(b,i,j) for b in bl])/2
    
def SzSz(bl,i,j):
    # return Sz(bl,i)@Sz(bl,j)
    return Sz(bl,i)*Sz(bl,j) # direct produce is sufficient since there're all diagonal
def SiSj(bl,i,j):
    return SpSm(bl,i,j)+SzSz(bl,i,j)

# good quantum number ============================
def ND(b):
    return bitcount(abs(b))
def NU(b,M):
    return M-ND(b)
def SZ(b,M):
    return M/2-ND(b)
# gen S2 matrix under a (M,Sz) subspace ======
def S2(bl,M):
    Nb = len(bl)
    m = sp.diags(([SZ(bl[0],M)**2]*Nb)) # diagonal part
    for i,j in its.product(range(M),range(M)):
        m += SpSm(bl,i,j)
    return m

def gen_S2_trans(bl,M):
    S2va,S2ve = np.linalg.eigh(S2(bl,M).toarray())
    S2va = np.round(S2va).astype(int)
    S2set = np.sort(list(set(S2va)))
    S = np.round(np.sqrt(1+4*S2set)-1)/2
    
    idx = []
    for s in S2set:
        i,j = np.argwhere(S2va==s)[[0,-1],0]
        idx.append([i,j+1]) # j+1: for the convenience of indexing
    return S,idx,S2ve

sign_map = {1:'+',-1:'-'}
def analysis_S2(bl,M,prec=4,eps=1e-4,mode=0,fraction=False):
    '''
    print the basis transform  S2
    Parameters
    ----------
    bl : list
        basis.
    M : int
        sites.
    prec : float, optional
        show how many decimal place. The default is 4.
    eps : float, optional
        DESCRIPTION. The default is 1e-4.
    mode : 0 or 1, optional
        Different ways of printing the basis of bl. The default is 0.
    fraction : bool, optional
        Whether or not to find possible fraction expressions when printing the 
        basis transform. The default is False.

    Returns
    -------
    S : list
        good quantum number S.
    idx : list
        index information of blocks labeled by S. Convenience for 
        extracting blocks from a block-diagonal matrix.
    S2ve : matrix
        Eigenvectors of S2 matrix.

    '''
    print('Sz= %g ==================='%SZ(bl[0],M))
    Nb = len(bl)
    S,idx,S2ve = gen_S2_trans(bl,M)
    blh = [tobit(b,M) for b in bl]
    syml = toFractional(S2ve,remove_sign=True)
    cnt = 0
    for ib in range(Nb):
        if cnt<len(idx) and ib==idx[cnt][0]:
            print('S= %g, size= %d ---------'%(S[cnt],idx[cnt][1]-idx[cnt][0]))
            cnt += 1
        ln = 'φ%d ='%ib
        for jb in range(Nb):
            val = abs(S2ve[jb,ib])
            if val<eps: continue
            sgn = sign_map[np.sign(S2ve[jb,ib])]
            if mode==0:
                if fraction: ln += f' {sgn}{syml[jb,ib]}|{blh[jb]}>'
                else: ln += f' {sgn}{val:.{prec}f}|{blh[jb]}>'
            elif mode==1:
                if fraction: ln += f' {sgn}{syml[jb,ib]}ϕ{jb}'
                else: ln += f' {sgn}{val:.{prec}f}ϕ{jb}'
        print(ln)
    print()
    return S,idx,S2ve

