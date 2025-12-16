import sys
sys.path.append("C:/Users/maxim/Documents/GitHub/Few_body_operators/Codes/") ## Important to bind the modules

import numpy as np
import h5py as h5
import time 

import Modules.trotter as trott

##     ----------------------------------------------------------------------------------------
##     ----------------------------------------------------------------------------------------

class classMPS:
    '''
    '''
    def __init__(self, L, occupations=None):
        '''
        '''
        self.L = L
        self.BondDim = np.zeros(L+1, dtype=int)
        self.TN = "MPS"
        if type(occupations) == type(None):
                occupations = np.zeros(L)
                occupations[:int(L/2)] = 1
        MPS = {}
        for l in range(L): ## only works for initial product states
            MPS["B%s"%l] = np.zeros((1,2,1), dtype=complex)
            MPS["B%s"%l][0,int(occupations[l]),0] = 1 ## Start in the state given by occupations |1100010...> = |\psi_G>
            MPS["Lam%s"%l] = np.ones((1), dtype=complex)
            self.BondDim[l] = 1
        MPS["Lam%s"%(l+1)] = np.ones((1), dtype=complex)
        self.BondDim[l+1] = 1
        self.MPS = MPS
        
        return
    
    def EE(self):
        for i in range(self.L):
            ss = self.MPS["Lam%s"%i]
            print(i, "Bond dim:", ss.shape, "S =",np.sum(-np.log(ss**2)*ss**2))
            
    def copy(self):
        A = classMPS(self.L)
        A.MPS = self.MPS.copy()
        A.BondDim = self.BondDim.copy()
        A.L = self.L
        
        return A

##     ----------------------------------------------------------------------------------------
##     ----------------------------------------------------------------------------------------    
    
class classMPO:
    '''
    '''
    def __init__(self, L, start="Id"):
        '''
        '''
        self.L = L
        self.BondDim = np.zeros(L+1, dtype=int)
        self.TN = "MPO"

        MPO = {}
        for l in range(L): ## only works for initial product states
            MPO["B%s"%l] = np.zeros((1,2,2,1), dtype=complex)
            MPO["B%s"%l][0,0,0,0] = 1/np.sqrt(2) ## Start in Id
            MPO["B%s"%l][0,1,1,0] = 1/np.sqrt(2) ## Start in Id
            MPO["Lam%s"%l] = np.ones((1), dtype=complex)
            self.BondDim[l] = 1
            
        MPO["Lam%s"%(l+1)] = np.ones((1), dtype=float)
        self.BondDim[l+1] = 1
        self.MPO = MPO
        
        return    
    
    def toMPS(self, majorana=False):
        vecO = classMPS(2*L)
        MPS = {}
        for l in range(self.L):
            A = self.MPO["B%s"%l].copy()
            lsize, rsize = A.shape[0], A.shape[-1]

            if majorana:
                gate = np.array(((1,0,0,1),(0,1,1j,0),(0,1,-1j,0),(1,0,0,-1)), dtype=complex) / np.sqrt(2)
                A = np.tensordot(gate.reshape(2,2,2,2), A, axes=([2,3],[1,2])).transpose([2,0,1,3])
            
            B = np.tensordot(np.diag(self.MPO["Lam%s"%l]), A, axes=([1],[0]))
            U, S, V = np.linalg.svd(B.reshape(lsize*2, rsize*2),full_matrices=False)
                        
            mask = S/np.linalg.norm(S) > 1e-10
            S = S[mask]
            S = S / np.linalg.norm(S)

            B2 = V[mask,:].reshape(-1, 2, rsize)
            MPS["B%s"%(2*l+1)]   = B2
            MPS["B%s"%(2*l)]     = np.tensordot(A, B2.conj(), axes=([2,3],[1,2])) 
            MPS["Lam%s"%(2*l+1)] = S
            MPS["Lam%s"%(2*l)]   = self.MPO["Lam%s"%l]
            vecO.BondDim[2*l]    = self.BondDim[l]
            vecO.BondDim[2*l+1]  = S.shape[0]
            
            if majorana:
                sz = np.array(((1,0),(0,-1)), dtype=complex)
                MPS["B%s"%(2*l+1)] = np.tensordot(sz, MPS["B%s"%(2*l+1)], axes=([1],[1])).transpose([1,0,2])

        vecO.MPS = MPS 
        
        return vecO

    def copy(self):
        A = classMPO(self.L)
        A.MPO = self.MPO.copy()
        A.BondDim = self.BondDim.copy()
        A.L = self.L
        
        return A
    
    def EE(self):
        for i in range(self.L):
            ss = self.MPO["Lam%s"%i]
            ss = ss[ss**2>1e-15]
            print(i, "Bond dim:", ss.shape, "S =",np.sum(-np.log(ss**2)*ss**2))
    
    def rotate(self, R, rot):
        active = np.arange(L)
        for i in range(self.L):
            l = self.L - i
            sect = active[:l]

            Rev, Revecs = np.linalg.eigh(R[:l,:l])        ## Compute the correlation matrix spectrum in the current set of orbitals. Remove the orbital that was just rotated from Q
            # order = np.argsort(Rev*(1-Rev))[::-1]       ## Sort in decreasing order from 1/2: last orbital has occupation the closest to 0 or 1
            order = np.argsort(0.5-np.abs(Rev))[::-1]     ## Sort in decreasing order from 1/2: last orbital has occupation the closest to 0 or 1
            Rev, Revecs = Rev[order], Revecs[:,order]

            ## Apply givens rotations to localize the last orbital and put it to the right of the active part. Stack orbitals from right to left with occupancies getting closer to 1/2.
            indices, givens = Givens_rotations(Revecs, [l-1], sect, direction="right")  

            for m, ind in enumerate(indices):
                gate_giv = np.eye(4, dtype=complex) ## Transforms the givens rotation in a quantum gate
                gate_giv[1:-1,1:-1] = givens[m].T ## why not conj ??
                gate_giv = gate_giv.reshape(2,2,2,2)

                Apply_gate_MPO(self, gate_giv, ind[0], ind[1], bothsides=True)

            ## Rotate the correlation matrix according to the rotation done by the set of givens rotations.
            rot = RotFromGivens(indices, givens, sect)
            R[:l,:l] = rot @ R[:l,:l] @ rot.conj().T
                    
            self.compress()
            
    def compress(self):
        for i in range(1,self.L):
            s = self.MPO["Lam%s"%i].copy()
            mask = s**2 > 1e-15
            self.MPO["Lam%s"%i] = self.MPO["Lam%s"%i][mask]
            self.MPO["B%s"%(i-1)] = self.MPO["B%s"%(i-1)][:,:,:,mask]
            self.MPO["B%s"%(i)] = self.MPO["B%s"%(i)][mask,:,:,:]
            
            self.BondDim[i] = s.shape[0]

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- Givens rotations  -----------------------------

def Givens_rotations(mat, loc, sect=None, direction="left"):
    '''
        Compute the givens rotations used to localize one or more orbitals on the MPS. 
        
        ----------
        Inputs: 
        ----------
        mat: transfer matrix: rows: current set of orbitals. columns: desired set of orbitals.
        loc: which column in mat corresponds to the orbital that we want to localize.
        sect: which orbitals will be affected by the rotation.
        direction: "left": localize the orbital on the left of sect. +---- "right": localize the orbital on the right. ----+
        
        ----------
        Returns: 
        ----------
        indices: list of the sites on which the local givens rotations are applied. shape: Ngates*2, type: int
        givens: list of the givens rotations (2*2 matrices) to be applied. shape: Ngates*2*2, type: complex
        '''
        
    givens = np.zeros((0,2,2), dtype=complex)
    indices = np.zeros((0,2), dtype=int)
    
    M = mat.copy()
    if direction == "right": 
        loc = loc[::-1]
    for n, k in enumerate(loc):
        if direction == "right":
            reduction = np.arange(sect[0], sect[-1]-n)
        else:
            reduction = np.arange(sect[-1]-1, sect[0]+n-1, -1)
        i_n = []
        g_n = []
        v = M[:,k].copy()
        for j in reduction: ## Stack the localized orbitals at the beginning
            q, p = v[j], v[j+1]

            norm = np.sqrt(np.abs(p)**2 + np.abs(q)**2)
            if norm < 1e-14:
                g = np.eye(2, dtype=complex)
            else:
                if direction == "left":
                    g = np.array(((q.conj(),p.conj()),(-p, q)), dtype=complex) / norm
                    v[j], v[j+1] = norm, 0 ## eliminate component j
                else:
                    g = np.array(((p,-q.conj()),(q, p.conj())), dtype=complex) / norm
                    v[j], v[j+1] = 0, norm ## eliminate component j

            i_n.append([j,j+1])
            g_n.append(g)
       
        ## Adapt the basis to what the layer of givens just changed
        rot = RotFromGivens(i_n, g_n, sect)
        M = rot @ M 
        
        if len(i_n)>0:
            indices = np.append(indices, np.array(i_n), axis=0)
            givens = np.append(givens, np.array(g_n), axis=0)
   
    return indices, givens

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def RotFromGivens(indices, givens, sect):
    '''
        Computes the actual rotation that is applied on the frame after having applied a set of givens rotations on the MPS. One orbital is rotated as we want, but the others are affected by this rotation (mostly a mess to keep orthonormality), so we want to keep track of it.
    
        ----------
        Inputs: 
        ----------
        indices: list of sites (i_n, i_n+1) on which gate g_n is applied. shape: N_G * 2, type:int
        givens: list of gates performing the givens rotation. Each element represents g_n. shape: N_G * 2 * 2, type:complex
        sect: list of the sites affected by the givens rotation. Number of sites gives the dimension of the rot. list of indices should coincide with the row/columns of row: if we rotate site 12-16, shift indices to 0-4 to be correctly applied on rot.
        
        ----------
        Returns: 
        ----------
        rot: rotation on the subspace. shape: len(sect) * len(sect), type=complex
    '''
    rot = np.eye(len(sect), dtype=complex)
    ## reverse order compared to what was applied on the MPS. 
    indices = indices[::-1]
    givens  = givens[::-1]
    for m, ind in enumerate(indices):
        rot[:, ind] = rot[:, ind] @ givens[m]
    
    return rot

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def Apply_Fermionic_Op(A, Op, l, side="L"):
    '''
    Applies a fermionic operator A (c or c+) on the tensor B_i of a symmetric MPO O.
    '''
    O = Op.copy()

    c = np.array(((0,1.),(0,0)),dtype=complex)
    sz = np.array(((1.,0),(0,-1.)),dtype=complex)

    if Op.TN == "MPO":
        wh  = 1 if side == "L" else 0
        wh2 = 1 if side == "L" else 2
        tr = [1,0,2,3] if side == "L" else [1,2,0,3]
        if A == "c":
            site_sign = np.arange(l)  ## J-W string
            for k in site_sign:
                O.MPO['B%s'%k] = np.tensordot(sz, O.MPO['B%s'%k], axes=([wh],[wh2])).transpose(tr)
            O.MPO['B%s'%l] = np.tensordot(c, O.MPO['B%s'%l], axes=([wh],[wh2])).transpose(tr)
        elif A == "n":
            n = (np.eye(2,dtype=complex) - sz) * 0.5
            O.MPO['B%s'%l] = np.tensordot(n, O.MPO['B%s'%l], axes=([wh],[wh2])).transpose(tr)
    
    elif Op.TN == "MPS":
        tr = [1,0,2]
        if (A == "c") or (A == "c+"):
            site_sign = np.arange(l)  ## J-W string
            for k in site_sign:
                O.MPS['B%s'%k] = np.tensordot(sz, O.MPS['B%s'%k], axes=([1],[1])).transpose(tr)
            c = c if A == "c" else c.T
            O.MPS['B%s'%l] = np.tensordot(c, O.MPS['B%s'%l], axes=([1],[1])).transpose(tr)
        elif A == "n":
            n = (np.eye(2,dtype=complex) - sz) * 0.5 
            O.MPS['B%s'%l] = np.tensordot(n, O.MPS['B%s'%l], axes=([1],[1])).transpose(tr)
        elif A == "nn":
            nn = np.zeros((4,4), dtype=complex)
            nn[0,0],nn[0,3],nn[3,0],nn[3,3] = 1, -1, -1, 1

            nn = nn.reshape(2,2,2,2) / 2
            
            Apply_gate_MPS(O, nn, l, l+1)
            
    return O

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def Trace(A, B, conjA = True):
    A = A.copy()
    B = B.copy()
    if A.TN == "MPO":
        if conjA:
            Pac = np.tensordot(A.MPO["B0"].conj(), B.MPO["B0"], axes=([0,1,2],[0,1,2])) 
        else:
            Pac = np.tensordot(A.MPO["B0"], B.MPO["B0"], axes=([0,2,1],[0,1,2])) 
        for i in range(1,L-1):    
            if conjA:
                Pac = np.tensordot(Pac, A.MPO["B%s"%i].conj(), axes=([0],[0]))
                Pac = np.tensordot(Pac, B.MPO["B%s"%i], axes=([0,1,2],[0,1,2]))
            else:
                Pac = np.tensordot(Pac, A.MPO["B%s"%i], axes=([0],[0]))
                Pac = np.tensordot(Pac, B.MPO["B%s"%i], axes=([0,2,1],[0,1,2]))
        if conjA:
            End = np.tensordot(A.MPO["B%s"%(i+1)].conj(), B.MPO["B%s"%(i+1)], axes=([1,2,3],[1,2,3]))
        else:
            End = np.tensordot(A.MPO["B%s"%(i+1)], B.MPO["B%s"%(i+1)], axes=([1,2,3],[1,2,3]))
    
    elif A.TN == "MPS":
        ## Always conjA
        Pac = np.tensordot(A.MPS["B0"].conj(), B.MPS["B0"], axes=([0,1],[0,1])) 
        for i in range(1,L-1):    
            Pac = np.tensordot(Pac, A.MPS["B%s"%i].conj(), axes=([0],[0]))
            Pac = np.tensordot(Pac, B.MPS["B%s"%i], axes=([0,1],[0,1]))
            
        End = np.tensordot(A.MPS["B%s"%(i+1)].conj(), B.MPS["B%s"%(i+1)], axes=([1,2],[1,2]))
    
    return  np.tensordot(Pac, End, axes=([0,1], [0,1]))


##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def MPO_Correlation_Matrix(Op):    
    O = Op.copy()
    norm_O = np.abs(Trace(O.copy(), O.copy(), conjA=True))
    R = np.zeros((2*L, 2*L), dtype=complex)
    for i in range(2*O.L):
            ind_i = i // 2
            side_i = "L" if i%2 == 0 else "R" ## Apply c on the left or on the right depending on j (direct or dual space)
            ### Apply <O|c+ or <O|c and c+|O> or c|O>
            O1 = Apply_Fermionic_Op("c", O.copy(), ind_i, side_i)
            for j in range(i,2*O.L):
                ind_j = j // 2 
                side_j = "L" if j%2 == 0 else "R" ## Apply c on the left or on the right depending on j (direct or dual space)

                O2 = Apply_Fermionic_Op("c", O.copy(), ind_j, side_j)

                ## Compute (<O|c+)(c|O>) as Tr(O1+, O2)
                # R[O.L*(i%2) + ind_i, O.L*(j%2)+ind_j] = Trace(O1, O2, conjA=True) / norm_O
                R[i, j] = Trace(O1, O2, conjA=True) / norm_O

    R = R + R.conj().T - np.diag(R.diagonal())

    return R

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def vec_MPO_Covariance_Matrix(Psi):
    eta = np.array(((0,1),(1,0)),dtype=complex)
    sz = np.array(((1,0),(0,-1)),dtype=complex)
    
    norm = np.ones((1,1),dtype=complex)
    for i in range(Psi.L-1):
        norm = np.tensordot(norm,  Psi.MPS["B%s"%i].conj(), axes=([0],[0]))
        norm = np.tensordot(norm,  Psi.MPS["B%s"%i], axes=([0,1],[0,1]))
    norm = np.tensordot(norm,  Psi.MPS["B%s"%(i+1)].conj(), axes=([0],[0]))
    norm = np.abs(np.tensordot(norm,  Psi.MPS["B%s"%(i+1)], axes=([0,1,2],[0,1,2])))

    Q = np.zeros((Psi.L, Psi.L), dtype=complex)
    for i in range(Psi.L):        
        Bi = Psi.MPS["B%s"%i]
        ### Apply <O|eta_i and eta_j|O>
        Bic = np.tensordot(eta, Bi, axes=([1],[1])).transpose(1,0,2)
        Lam = np.diag(Psi.MPS["Lam%s"%(i)])

        ## For next element to compute
        Bc =  np.tensordot(Lam, Bic, axes=([1],[0]))
        B  =  np.tensordot(Lam, Bi,  axes=([1],[0]))
        
        Pac = np.tensordot(Bc.conj(), B, axes=([0,1],[0,1]))
        for j in range(i+1,Psi.L,1): ## Only the upper triangle
            Bj = Psi.MPS["B%s"%j]
            Bjc = np.tensordot(eta, Bj, axes=([1],[1])).transpose(1,0,2)
            
            A = np.tensordot(Pac, Bj.conj(), axes=([0],[0]))
            
            Q[i,j] = 2 * np.tensordot(A, Bjc, axes=([0,1,2],[0,1,2]))
                            
            ## Propagate with fermionic signs for next step
            Bjz = np.tensordot(sz, Bj, axes=([0],[1])).transpose(1,0,2)
            Pac = np.tensordot(Pac, Bjz.conj(), axes=([0],[0])) ## some operations can be done less times
            Pac = np.tensordot(Pac, Bj, axes=([0,1],[0,1]))
    
    Q = Q / norm
    Q = 1j * (Q - Q.T - np.diag(np.diagonal(Q))) #/2 # if majo are not normalized

    return Q

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def Correlation_Matrix_vectorized_MPO(Psi): ## Does not work for MPOs
    c = np.array(((0,1),(0,0)),dtype=complex)
    sz = np.array(((1,0),(0,-1)),dtype=complex)

    Q = np.zeros((Psi.L,Psi.L), dtype=complex)
    PacA = np.ones((1,1), dtype=complex)
    for i in range(Psi.L): ## even sites: c+, odd: c
        Bi = Psi.MPS["B%s"%i]

        ### Apply <O|c+ and c|O>
        whi = 1 if (i%2) == 0 else 0

        Bic = np.tensordot(c, Bi, axes=([whi],[1])).transpose(1,0,2)

        Lam = np.diag(Psi.MPS["Lam%s"%(i)])

        A = np.tensordot(Lam, Bic, axes=([1],[0]))
        Q[i,i] = np.tensordot(A.conj(), A, axes=([0,1,2],[0,1,2]))

        PacB = np.tensordot(PacA, Bic.conj(), axes=([0],[0])) ## some operations can be done less times
        PacB = np.tensordot(PacB, Bi, axes=([0,1],[0,1]))

        Biz = np.tensordot(sz, Bi, axes=([whi],[1])).transpose(1,0,2)
        PacA = np.tensordot(PacA, Biz.conj(), axes=([0],[0])) ## some operations can be done less times
        PacA = np.tensordot(PacA, Bi, axes=([0,1],[0,1]))

        ## For next element to compute
        Bc =  np.tensordot(Lam, Bic, axes=([1],[0]))
        B  =  np.tensordot(Lam, Bi,  axes=([1],[0]))

        Pac = np.tensordot(Bc.conj(), B, axes=([0,1],[0,1]))
        cpt = 1
        for j in range(i+1,Psi.L,1): ## Only the upper triangle
            Bj = Psi.MPS["B%s"%j]

            whj = 1 if (j%2) == 0 else 0
            Bjc = np.tensordot(c, Bj, axes=([whj],[1])).transpose(1,0,2)

            if (i%2) == (j%2):
                A = np.tensordot(Pac, Bj.conj(), axes=([0],[0]))
                Q[i,j] = np.tensordot(A, Bjc, axes=([0,1,2],[0,1,2]))
            else:
                A = np.tensordot(PacB, Bj.conj(), axes=([0],[0]))
                Q[i,j] = np.tensordot(A, Bjc, axes=([0,1,2],[0,1,2]))

            ## Propagate with fermionic signs for next step
            if (cpt%2) == 0: 
                Bjz = np.tensordot(sz, Bj, axes=([whj],[1])).transpose(1,0,2)
            else:
                Bjz = Bj

            Pac = np.tensordot(Pac, Bjz.conj(), axes=([0],[0])) ## some operations can be done less times
            Pac = np.tensordot(Pac, Bj, axes=([0,1],[0,1]))

            if (cpt%2) == 1: 
                Bjz = np.tensordot(sz, Bj, axes=([whj],[1])).transpose(1,0,2)
            else:
                Bjz = Bj
            PacB = np.tensordot(PacB, Bjz.conj(), axes=([0],[0])) ## some operations can be done less times
            PacB = np.tensordot(PacB, Bj, axes=([0,1],[0,1]))

            cpt += 1

    Q = Q + Q.conj().T - np.diag(np.diagonal(Q))

    return Q

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def MPS_Correlation_Matrix(Psi, sites=None): ## Does not work for MPOs
    '''
        Given a connected set of sites, compute the correlation matrix Q whose elements are given by Q_ij = \langle \psi | c^{\dagger}_{i}c^{}_{j} | \psi \rangle, where operators c_i is the current orbital at position i in the MPS.

        ----------
        Inputs: 
        ----------
        sites: sites on which to compute the correlation matrix.
        
        ----------
        Returns: 
        ----------
        Q: Correlation matrix. Shape: len(sites) * len(sites). Type: complex
    '''
    
    c = np.array(((0,1),(0,0)),dtype=complex)
    sz = np.array(((1,0),(0,-1)),dtype=complex)

    norm = np.ones((1,1),dtype=complex)
    for i in range(Psi.L-1):
        norm = np.tensordot(norm,  Psi.MPS["B%s"%i].conj(), axes=([0],[0]))
        norm = np.tensordot(norm,  Psi.MPS["B%s"%i], axes=([0,1],[0,1]))
    norm = np.tensordot(norm,  Psi.MPS["B%s"%(i+1)].conj(), axes=([0],[0]))
    norm = np.abs(np.tensordot(norm,  Psi.MPS["B%s"%(i+1)], axes=([0,1,2],[0,1,2])))

    if type(sites) == type(None):
        sites = np.arange(Psi.L)
        
    Q = np.zeros((sites.shape[0], sites.shape[0]), dtype=complex)
    for ind, i in enumerate(sites):
        Bi = Psi.MPS["B%s"%i]
        ### Apply <O|c+ and c|O>
        Bic = np.tensordot(c, Bi, axes=([1],[1])).transpose(1,0,2)
    
        Lam = np.diag(Psi.MPS["Lam%s"%(i)])

        A = np.tensordot(Lam, Bic, axes=([1],[0]))
        Q[i,i] = np.tensordot(A.conj(), A, axes=([0,1,2],[0,1,2]))
        
        ## For next element to compute
        Bc =  np.tensordot(Lam, Bic, axes=([1],[0]))
        B  =  np.tensordot(Lam, Bi,  axes=([1],[0]))
        
        Pac = np.tensordot(Bc.conj(), B, axes=([0,1],[0,1]))
        for j in sites[ind+1:]: ## Only the upper triangle
            Bj = Psi.MPS["B%s"%j]
            Bjc = np.tensordot(c, Bj, axes=([1],[1])).transpose(1,0,2)
            
            A = np.tensordot(Pac, Bj.conj(), axes=([0],[0]))
            Q[i,j] = np.tensordot(A, Bjc, axes=([0,1,2],[0,1,2]))
                            
            ## Propagate with fermionic signs for next step
            Bjz = np.tensordot(sz, Bj, axes=([0],[1])).transpose(1,0,2)
            Pac = np.tensordot(Pac, Bjz.conj(), axes=([0],[0])) ## some operations can be done less times
            Pac = np.tensordot(Pac, Bj, axes=([0,1],[0,1]))
            
    Q = Q/norm
    Q = Q + Q.conj().T - np.diag(np.diagonal(Q))
    
    return Q

##      ----------------------------------------------------------------------
##      ----------------------------------------------------------------------

def Apply_gate_MPS(Psi, gate, l1, l2):
    '''
        In Place. Apply a 2-body gate on an MPS by keeping the MPS structure. 
        
        ----------
        Inputs: 
        ----------
        Psi: MPS on which the gate is applied. Modified in place.
        gate: gate that is to be applied. shape:(2,2,2,2) tensor, indices (\sigma_i, \sigma_{i+1},\sigma'_i, \sigma'_{i+1} )
        l1: first site on which the gate is applied. 
        l2: second site on which the gate is applied, only i+1 works, otherwise implement swaps.
    '''
    MPS = Psi.MPS
        
    lshape, rshape = MPS["B%s"%l1].shape[0], MPS["B%s"%l2].shape[-1]

    PhiB = np.tensordot(MPS["B%s"%l1], MPS["B%s"%l2], ([2],[0])) ## Legs correctly ordered
    
    ## UPhi = U * PhiB
    UPhiB = np.tensordot(gate, PhiB, ([2,3],[1,2]))
    UPhiB = np.transpose(UPhiB, [2,0,1,3]) ## Swap back indices
    
    ## Absorb the lambda
    Phi = np.tensordot(np.diag(MPS["Lam%s"%l1]), UPhiB, ([1],[0])).reshape(lshape * 2, rshape * 2)
    
    ## Splits the tensor back to an MPS
    U, S, Vh = np.linalg.svd(Phi, full_matrices=False)

    mask = (S / np.linalg.norm(S)) > 1e-10
    S = S[mask]
    S = S / np.linalg.norm(S)
    
    B2 = Vh[mask,:].reshape(-1, 2, rshape)

    B1 = np.tensordot(UPhiB, B2.conj(), ([2,3],[1,2]))

    Psi.MPS["B%s"%l1], Psi.MPS["Lam%s"%l2], Psi.MPS["B%s"%l2] = B1, S, B2
    Psi.BondDim[l2] = S.shape[0]
    
    return

##      ----------------------------------------------------------------------
##      ---------------------------------------------------- ------------------

def Apply_gate_MPO(Psi, gate, l1, l2, bothsides=True):
    '''
        In Place. Apply a 2-body gate on an MPO by keeping the MPO structure. 
        
        ----------
        Inputs: 
        ----------
        Psi: MPS on which the gate is applied. Modified in place.
        gate: gate that is to be applied. shape:(2,2,2,2) tensor, indices (\sigma_i, \sigma_{i+1},\sigma'_i, \sigma'_{i+1} )
        l1: first site on which the gate is applied. 
        l2: second site on which the gate is applied, only i+1 works, otherwise implement swaps.
    '''
    MPO = Psi.MPO
        
    ldim, rdim = MPO["B%s"%l1].shape[0], MPO["B%s"%l2].shape[-1]

    PhiB = np.tensordot(MPO["B%s"%l1], MPO["B%s"%l2], ([3],[0]))
    ## UPhi = U * PhiB
    UPhiB = np.tensordot(gate, PhiB, ([2,3],[1,3])).transpose([2,0,3,1,4,5])
    if bothsides:
        UPhiB = np.tensordot(gate.conj(), UPhiB, ([2,3],[2,4])).transpose([2,3,0,4,1,5])
    
    ## Absorb the lambda
    Phi = np.tensordot(np.diag(MPO["Lam%s"%l1]), UPhiB, ([1],[0])).reshape(ldim * 2 * 2, rdim * 2 * 2)
        
    ## Splits the tensor back to an MPS
    U, S, Vh = np.linalg.svd(Phi, full_matrices=False)

    mask = (S / np.linalg.norm(S)) > 1e-10
    S = S[mask]
    S = S / np.linalg.norm(S)
    
    B2 = Vh[mask,:].reshape(-1, 2, 2, rdim)
     
    B1 = np.tensordot(UPhiB, B2.conj(), ([3,4,5],[1,2,3]))

    Psi.MPO["B%s"%l1], Psi.MPO["Lam%s"%l2], Psi.MPO["B%s"%l2] = B1, S, B2
    Psi.BondDim[l2] = S.shape[0]
    
    return

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- Hamiltonian Parameters -----------------------------

def givens_to_MPO(g, ind, L):
    Id = np.array(((1,0),(0, 1)), dtype=complex)
    sz = np.array(((1,0),(0,-1)), dtype=complex)
    sp = np.array(((0,0),(1, 0)), dtype=complex)
    sm = np.array(((0,1),(0, 0)), dtype=complex)

    # if (ind[0]%2)==0:
    #     sp = np.array(((0,0),(1, 0)), dtype=complex)
    #     sm = np.array(((0,1),(0, 0)), dtype=complex)
    # else:
    #     sp = np.array(((0,1),(0, 0)), dtype=complex)
    #     sm = np.array(((0,0),(1, 0)), dtype=complex)

    MPO_G = classMPO(L)
    if ind[0] > 0:
        MPO_G.MPO['B0'] = np.zeros((1,2,2,2), dtype=complex)
        MPO_G.MPO['B0'][0,:,:,0] = Id
        MPO_G.MPO['B0'][0,:,:,1] = sz
        
        # MPO_G.MPO['B0'] = np.zeros((1,2,2,1), dtype=complex)
        # MPO_G.MPO['B0'][0,:,:,0] = sz

        MPO_G.BondDim[0] = 1
    for i in range(1, ind[0]):
        MPO_G.MPO['B%s'%i] = np.zeros((2,2,2,2), dtype=complex)
        MPO_G.MPO['B%s'%i][0,:,:,0] = Id
        MPO_G.MPO['B%s'%i][1,:,:,1] = sz
        
        # MPO_G.MPO['B%s'%i] = np.zeros((1,2,2,1), dtype=complex)
        # MPO_G.MPO['B%s'%i][0,:,:,0] = sz

        MPO_G.BondDim[i] = 2

    a,b,c,d = g[0,0], g[0,1], g[1,0], g[1,1]

    MPO_G.MPO['B%s'%ind[0]] = np.zeros((2,2,2,4), dtype=complex)
    MPO_G.MPO['B%s'%ind[0]][1,:,:,1] = b * sp
    MPO_G.MPO['B%s'%ind[0]][1,:,:,3] = c * sm
    # MPO_G.MPO['B%s'%ind[0]] = np.zeros((1,2,2,4), dtype=complex)
    # MPO_G.MPO['B%s'%ind[0]][0,:,:,1] = g[0,1] * sp
    # MPO_G.MPO['B%s'%ind[0]][0,:,:,3] = g[1,0] * sp
    MPO_G.MPO['B%s'%ind[0]][0,:,:,0] = Id
    MPO_G.MPO['B%s'%ind[0]][0,:,:,2] = 0.5 * (Id-sz)
    
    MPO_G.MPO['B%s'%ind[1]] = np.zeros((4,2,2,1), dtype=complex)
    MPO_G.MPO['B%s'%ind[1]][0,:,:,0] = Id + (d-1) * 0.5 * (Id-sz)
    MPO_G.MPO['B%s'%ind[1]][1,:,:,0] = sm
    MPO_G.MPO['B%s'%ind[1]][2,:,:,0] = (a-1) * Id + (2-a-d) * 0.5 * (Id-sz)
    MPO_G.MPO['B%s'%ind[1]][3,:,:,0] = sp
    
    MPO_G.BondDim[ind[0]] = 2
    MPO_G.BondDim[ind[1]] = 4

    for i in range(ind[1]+1, L):
        MPO_G.MPO['B%s'%i] = np.zeros((1,2,2,1), dtype=complex)
        MPO_G.MPO['B%s'%i][0,:,:,0] = Id
        MPO_G.BondDim[i] = 1

    return MPO_G

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- Hamiltonian Parameters -----------------------------

def apply_MPO_MPS(O, Psi):
    Res = classMPS(Psi.L)
    
    Res.MPS["B0"] = np.tensordot(O.MPO["B0"], Psi.MPS["B0"], axes=([2],[1])).transpose([0,3,1,2,4]).reshape(1,2,O.MPO["B0"].shape[3],Psi.MPS["B0"].shape[2])
    Res.MPS["Lam0"] = np.ones((1), dtype=complex)

    for i in range(1,Psi.L):
        A = np.tensordot(Res.MPS["B%s"%(i-1)], O.MPO["B%s"%i], axes=([2],[0]))
        A = np.tensordot(A, Psi.MPS["B%s"%i], axes=([2,3],[0,1]))
        
        
        ldim, rdim = A.shape[0], A.shape[3]*A.shape[4]

        B = np.tensordot(np.diag(Res.MPS["Lam%s"%(i-1)]), A, axes=([1],[0]))
        B[np.abs(B)<1e-20] = 0
        
        _, S, V = np.linalg.svd(B.reshape(ldim * 2, 2 * rdim), full_matrices=False)
                
        mask = S / np.linalg.norm(S) > 1e-10
        
        S = S[mask]
        S = S / np.linalg.norm(S)
        
        B2 = V[mask,:].reshape(-1,2,O.MPO["B%s"%i].shape[3], Psi.MPS["B%s"%i].shape[2])
        B1 = np.tensordot(A, B2.conj(), axes=([2,3,4],[1,2,3]))

        Res.MPS["B%s"%(i-1)], Res.MPS["Lam%s"%i], Res.MPS["B%s"%i] = B1, S, B2
        Res.BondDim[i] = S.shape[0]
        
    Res.MPS["B%s"%i] = Res.MPS["B%s"%i].reshape(Res.MPS["B%s"%i].shape[0],2,1)
    
    return Res


## 
L = 10 ## Size of the tensor network
d = 2     ## Local physical dimension

model = "IRLM"     ## "Heis_nn" or "IRLM"
save = False     ## Save the data or not

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- TN Parameters --------------------------------------
phys_dims     = 1         ## Number of physical dimensions for each tensor. = 1 if 1 tensor = 1 site: 
alpha            = -1         ## Definition of the super-charge
truncation_type = "global"     ## can be: "global", "block_threshold" or  "block" or "adaptative_block_threshold"
chi_max         = 1024        ## Maximum bond dimension
s = int(L/2) if alpha == L+1 else L ## Operator charge according to the super-charge Q_\alpha

trunc_sing_vals = 1e-10

## singular values discarded below this threshold.
data_as_tensors = False     ## Stores the data as tensors (True) or as matrices (False). The latter is faster, but works only in 1d.

##     ----------------------------------------------------------------------------------------
##     ------------------------------- T. evol / Trotter Parameters ---------------------------
TrottOrder      = 4    ## Order of trotter decomposition (1,2,4)
dt = DT         = 0.01    ## time step
T               = 0.4   ## Final time

nStepsCalc      = 100     ## We compute a physical observable every nStepsCalc

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- Quantities we want to store  ---------------------------
Nsteps = int(T/dt) ## Steps of time evolution
nComp = (Nsteps//nStepsCalc + Nsteps%nStepsCalc) ## Nb of elements that we compute (every nStepsCalc)

Rev_t = np.zeros((2*L,nComp), dtype=float)     ## Stores the eigenvalues of the R matrix
CumErr_t = np.zeros((L,nComp), dtype=float)     ## Stores the cumulative error
AbsTime  = np.zeros(nComp, dtype=float)     ## Stores the time
Norm     = np.zeros(nComp, dtype=float)     ## Stores the norm

##     ----------------------------------------------------------------------------------------
##     ----------------------------------- Parameters -------------------------------
params = {'L':L, 'dt':dt, 'alpha':alpha, 'Truncation':truncation_type, 'chi': chi_max, 'Sect':s, 'TrottOrder':TrottOrder, 'Nsteps':Nsteps, 'phys_dims':phys_dims}
if model == 'Heis_nn':
    J  = 1
    Jz = 0
    params.update({'d':d, 'J':J, 'Jz':Jz, 'Uint':0, 'V':0, 'gamma':0, 'periodT':0, 'hmean':0, 'hdrive':0})
    seq0, seq1 = [np.array((i,i+1)) for i in range(0,L-1,2)], [np.array((i,i+1)) for i in range(1,L-1,2)]
    gates_layers = {"H0":seq0, "H1":seq1}     ## Sequences of gates that are applied at each Trott. step

elif model == 'IRLM':
    Uint   = 0.0 ## U is already the name of the gate
    V      = 0.2
    gamma  = 0.5
    ed     = 0
    params.update({'L':L, 'd':d, 'Uint':Uint, 'V':V, 'gamma':gamma, 'ed':ed, 'J':0, 'Jz':0})
    seq0, seq1, seq2 = [np.array((0,1))], [np.array((i,i+1)) for i in range(2,L-1,2)], [np.array((i,i+1)) for i in range(1,L-1,2)]
    gates_layers = {"H0":seq0, "H1":seq1, "H2":seq2}     ## Sequences of gates that are applied at each Trott. step

[print(key, ":", params[key]) for key in params.keys()]
print("\n")

nParts = len(gates_layers.keys()) ## Nparts is the number of commuting layers
TrotterSteps, U, _, _ = trott.Trotter_Seq(TrottOrder, Nsteps, dt, nParts, nStepsCalc, params, alpha=alpha, data_as_tensors=data_as_tensors, model=model, Hsteps = list(gates_layers.keys()), symm = False)

##     ----------------------------------------------------------------------------------------
##     -------------------------------    Saving data ------------------------

filename = "../data/MPO/%s/U_%s"%(model,model)
for key in params.keys():
        filename = filename + str("_"+key+str(params[key]))
if save:
        f =  h5.File(filename+".h5", "w")

##     ----------------------------------------------------------------------------------------
##     -------------------------------     Initialize O(0) ------------------------

Op = classMPO(L, "Id")
# sz = np.array(((1,0),(0,-1)), dtype=complex)
# Op.MPO["B0"] = np.tensordot(sz, Op.MPO["B0"], axes=([1],[1])).transpose([1,0,2,3]à)
# Op.BondDim()

##     ----------------------------------------------------------------------------------------
##     --------------------------------    Time evolution w/ TEBD    --------------------------

t, cpt, step_t0 = 0.0, 0, 0
st = time.time()
for step, dt, newTstep, ComputeObs in TrotterSteps[step_t0:]:
    ## Apply the gates for the given layer.
    for (l1,l2) in gates_layers[step]:
        Apply_gate_MPO(Op, U[step, dt], l1, l2, bothsides=False)

    ## Real time step dt done
    if newTstep:
        t += DT
        print(Op.BondDim)
        print("Evolution time: ", np.round(t,4), "Computational time: ", time.time()-st, "\n")
        st = time.time()

##     ----------------------------------------------------------------------------------------
##     --------------------------------    Rotation   -----------------------------------------

R = MPO_Correlation_Matrix(Op)
Rev, Revecs = np.linalg.eigh(R)

A = R[::2,1::2].copy()
Aev, Aevecs = np.linalg.eigh(A)

print("A eigenvalues: ", 0.5-np.abs(Aev))
print("R eigenvalues: ", Rev)

## Trace with MPO
Uni = Apply_Fermionic_Op("n", Op, 0, "R")
niU = Apply_Fermionic_Op("n", Op, 0, "L")
Sz_ex = Trace(niU, Uni)
print("\nTr(U+(t) ni U(t) ni) = ", Sz_ex )

## Trace with vectorized MPO into MPS
# vec_niU = Apply_Fermionic_Op("n", vecO, 0)
# vec_Uni = Apply_Fermionic_Op("n", vecO, 1)
# print("\nErr vec: ", 1 - Trace(vec_niU, vec_Uni) / Sz_ex )

occ = Rev>0.5
D = Revecs[:,occ]

Sz = np.sum( D[0,:].conj() * D[0,:]) - np.sum(D[0,:,None].conj() * D[0,:,None] * D[1,None,:].conj() * D[1,None,:]) + np.sum(D[0,:,None].conj() * D[1,:,None] * D[1,None,:].conj() * D[0,None,:])
print("Err rot Wick: ", 1 - Sz / Sz_ex)

print("Before rotation: \n", Op.BondDim)

## Rotate orbitals one by one in the MPS
Op.rotate(A, Aevecs)
Op.EE()
print("After rotation: \n", Op.BondDim)

print(stop)

print("Diagonal of real R: ", np.abs(R.diagonal()))

A = np.zeros(vecO.L**2, dtype=object)
Ri_o = np.zeros(vecO.L**2, dtype=complex) ## left  side: 2i
Ri_e = np.zeros(vecO.L**2, dtype=complex) ## right side: 2i+1
for i in range(vecO.L):
    cU = Apply_Fermionic_Op('c+', vecO, i)
    for j in range(vecO.L):
        ind = i*vecO.L + j
        A[ind] = Apply_Fermionic_Op('c', cU, j)
        Ri_e[ind] = full_rot[0,i].conj() * full_rot[0,j] ## 0 since we want n_0
        Ri_o[ind] = full_rot[1,i].conj() * full_rot[1,j]

Sz = 0
for a in range(vecO.L**2):
    print("Wick: ", a) if (a%10)==0 else None
    for b in range(vecO.L**2):
        Sz += Ri_e[a].conj() * Ri_o[b] * Trace(A[a], A[b])

print("Sz after rot:", 1 - Sz / Sz_ex)

print(stop)

# Q = Correlation_Matrix(vecO, sites = np.arange(2*L))

if save:
    f.create_dataset("Rev_t",       data=Rev_t)
    f.create_dataset("CumErr_t",    data=CumErr_t)
    f.create_dataset("t",           data=AbsTime)
    f.create_dataset("Norm",        data=Norm)
    f.close()
