from ED_spin import *
import numpy as np
from scipy.linalg import block_diag


# example 1: 2sites
M = 2
Bl = build_full_basis(M)
Bl = [b for bl in Bl for b in bl]

H = SiSj(Bl,0,1)
print('H=\n',H.toarray())


#%% example 2: 2sites, but expand in each subspace (M,Sz,S)
M = 2
Bl = build_full_basis(M)


for bl in Bl:
    H = SiSj(bl,0,1)

    S,idx,S2ve = analysis_S2(bl, M, mode=0,fraction=True)
    
    print('H=',H.toarray())
    HS2 = S2ve.T.conj()@H@S2ve
    print('HS2=',HS2)
    
    print('\n')
    
#%% example 3: 3sites
Hfull = []

M = 3
Bl = build_full_basis(M)


bonds = [[0,1],[1,2],[2,0]]
Nbonds = len(bonds)
Js = [1,1,-1.3]


vas = []

for bl in Bl:
    H = SS_ij(bl,bonds,Js)
    H = H.toarray()
    
    Hfull.append(H)
    
    va,ve = np.linalg.eigh(H)
    vas.append(va)

    S,idx,S2ve = analysis_S2(bl, M, mode=0,fraction=True)
    
    print('H=',H)
    
    HS2 = S2ve.T.conj()@H@S2ve
    print('HS2=',HS2)
      
vas = np.sort(np.hstack(vas))
print(vas)
    
Hf = block_diag(*Hf)
#%% compare
def kron(*args):
    m = 1
    for v in args:
        m = np.kron(m,v)
    return m

s0 = np.array([[1,0],[0,1]])
sx = np.array([[0,1],[1,0]])/2
sy = np.array([[0,-1j],[1j,0]])/2
sz = np.array([[1,0],[0,-1]])/2

Hp = kron(sx,sx,s0)+kron(sy,sy,s0)+kron(sz,sz,s0)  \
    +kron(s0,sx,sx)+kron(s0,sy,sy)+kron(s0,sz,sz)  \
    -1.3*(kron(sx,s0,sx)+kron(sy,s0,sy)+kron(sz,s0,sz) )
    
vap,vep = np.linalg.eigh(Hp)
print(vap)

