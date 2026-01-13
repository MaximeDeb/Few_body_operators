import sys
sys.path.append("C:/Users/maxim/Documents/GitHub/Few_body_operators/Codes/")

import numpy as np
import h5py as h5
import time
from typing import Optional, Dict, List, Tuple

import Modules.trotter as trott


class classMPS:
    """Matrix Product State representation."""
    
    def __init__(self, L: int, occupations: Optional[np.ndarray] = None):
        """
        Initialize MPS.
        
        Args:
            L: System size
            occupations: Initial occupation pattern (default: half-filled)
        """
        self.L = L
        self.BondDim = np.zeros(L + 1, dtype=int)
        self.TN = "MPS"
        
        if occupations is None:
            occupations = np.zeros(L)
            occupations[:L // 2] = 1
            
        self.MPS = {}
        for l in range(L):
            self.MPS[f"B{l}"] = np.zeros((1, 2, 1), dtype=complex)
            self.MPS[f"B{l}"][0, int(occupations[l]), 0] = 1
            self.MPS[f"Lam{l}"] = np.ones(1, dtype=complex)
            self.BondDim[l] = 1
            
        self.MPS[f"Lam{L}"] = np.ones(1, dtype=complex)
        self.BondDim[L] = 1
    
    def EE(self) -> None:
        """Print entanglement entropy for each bond."""
        for i in range(self.L):
            ss = self.MPS[f"Lam{i}"]
            ss_sq = ss**2
            mask = ss_sq > 1e-15
            S = -np.sum(np.log(ss_sq[mask]) * ss_sq[mask]) if mask.any() else 0
            print(f"{i} Bond dim: {ss.shape} S = {S}")
    
    def copy(self):
        """Create a deep copy of the MPS."""
        A = classMPS(self.L)
        A.MPS = {k: v.copy() for k, v in self.MPS.items()}
        A.BondDim = self.BondDim.copy()
        return A


class classMPO:
    """Matrix Product Operator representation."""
    
    def __init__(self, L: int, start: str = "Id"):
        """
        Initialize MPO.
        
        Args:
            L: System size
            start: Initial state (default: "Id" for identity)
        """
        self.L = L
        self.BondDim = np.zeros(L + 1, dtype=int)
        self.TN = "MPO"
        
        self.MPO = {}
        for l in range(L):
            self.MPO[f"B{l}"] = np.zeros((1, 2, 2, 1), dtype=complex)
            self.MPO[f"B{l}"][0, 0, 0, 0] = 1 / np.sqrt(2)
            self.MPO[f"B{l}"][0, 1, 1, 0] = 1 / np.sqrt(2)
            self.MPO[f"Lam{l}"] = np.ones(1, dtype=complex)
            self.BondDim[l] = 1
            
        self.MPO[f"Lam{L}"] = np.ones(1, dtype=float)
        self.BondDim[L] = 1
    
    def toMPS(self, majorana: bool = False):
        """Convert MPO to vectorized MPS representation."""
        vecO = classMPS(2 * self.L)
        MPS = {}
        
        for l in range(self.L):
            A = self.MPO[f"B{l}"].copy()
            lsize, rsize = A.shape[0], A.shape[-1]
            
            if majorana:
                gate = np.array([
                    [1, 0, 0, 1],
                    [0, 1, 1j, 0],
                    [0, 1, -1j, 0],
                    [1, 0, 0, -1]
                ], dtype=complex) / np.sqrt(2)
                A = np.tensordot(gate.reshape(2, 2, 2, 2), A, axes=([2, 3], [1, 2]))
                A = A.transpose([2, 0, 1, 3])
            
            B = np.tensordot(np.diag(self.MPO[f"Lam{l}"]), A, axes=([1], [0]))
            U, S, V = np.linalg.svd(B.reshape(lsize * 2, rsize * 2), full_matrices=False)
            
            mask = S / np.linalg.norm(S) > 1e-10
            S = S[mask] / np.linalg.norm(S[mask])
            
            B2 = V[mask, :].reshape(-1, 2, rsize)
            MPS[f"B{2*l+1}"] = B2
            MPS[f"B{2*l}"] = np.tensordot(A, B2.conj(), axes=([2, 3], [1, 2]))
            MPS[f"Lam{2*l+1}"] = S
            MPS[f"Lam{2*l}"] = self.MPO[f"Lam{l}"]
            
            vecO.BondDim[2 * l] = self.BondDim[l]
            vecO.BondDim[2 * l + 1] = S.shape[0]
            
            if majorana:
                sz = np.array([[1, 0], [0, -1]], dtype=complex)
                MPS[f"B{2*l+1}"] = np.tensordot(sz, MPS[f"B{2*l+1}"], axes=([1], [1]))
                MPS[f"B{2*l+1}"] = MPS[f"B{2*l+1}"].transpose([1, 0, 2])
        
        vecO.MPS = MPS
        return vecO
    
    def copy(self):
        """Create a deep copy of the MPO."""
        A = classMPO(self.L)
        A.MPO = {k: v.copy() for k, v in self.MPO.items()}
        A.BondDim = self.BondDim.copy()
        return A
    
    def EE(self) -> None:
        """Print entanglement entropy for each bond."""
        for i in range(self.L):
            ss = self.MPO[f"Lam{i}"]
            ss_sq = ss**2
            mask = ss_sq > 1e-15
            if mask.any():
                S = -np.sum(np.log(ss_sq[mask]) * ss_sq[mask])
                print(f"{i} Bond dim: {ss[mask].shape} S = {S}")
    
    def compress(self, th: float = 1e-12) -> None:
        """
        Compress MPO by removing small singular values.
        
        Args:
            th: Threshold for singular value truncation
        """
        for i in range(1, self.L):
            s = self.MPO[f"Lam{i}"]
            mask = s**2 > th
            
            if not mask.all():
                self.MPO[f"Lam{i}"] = s[mask]
                self.MPO[f"B{i-1}"] = self.MPO[f"B{i-1}"][:, :, :, mask]
                self.MPO[f"B{i}"] = self.MPO[f"B{i}"][mask, :, :, :]
                self.BondDim[i] = mask.sum()
    
    def rotate_Unit(self, R: np.ndarray) -> None:
        """
        Rotate MPO using unitary transformation.
        
        Args:
            R: Correlation matrix for rotation
        """
        active = np.arange(self.L)
        
        for i in range(self.L):
            print(f"Rotating i: {i}")
            l = self.L - i
            sect = active[:l]
            
            A = R[::2, 1::2].copy()
            U, S, Vd = np.linalg.svd(A[:l, :l])
            
            order = np.argsort(0.5 - np.abs(S))[::-1]
            S, Lev, Rev = S[order], U[:, order], Vd.conj().T[:, order]
            
            indices, Lgivens = Givens_rotations(Lev, [l - 1], sect, direction="right")
            indices, Rgivens = Givens_rotations(Rev.conj(), [l - 1], sect, direction="right")
            
            for m, ind in enumerate(indices):
                Lgate_giv = np.eye(4, dtype=complex)
                Rgate_giv = np.eye(4, dtype=complex)
                Lgate_giv[1:-1, 1:-1] = Lgivens[m].T
                Rgate_giv[1:-1, 1:-1] = Rgivens[m].T
                Lgate_giv = Lgate_giv.reshape(2, 2, 2, 2)
                Rgate_giv = Rgate_giv.reshape(2, 2, 2, 2)
                
                Apply_gate_MPO(self, Lgate_giv, ind[0], ind[1], 
                             bothsides=False, side="L", th=1e-12, right_gate=Rgate_giv)
            
            R = MPO_Correlation_Matrix(self)
            print(self.BondDim)
            self.compress(1e-15)


def Givens_rotations(mat: np.ndarray, loc: List[int], 
                    sect: Optional[np.ndarray] = None, 
                    direction: str = "left") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Givens rotations to localize orbitals.
    
    Args:
        mat: Transfer matrix
        loc: Columns corresponding to orbitals to localize
        sect: Orbitals affected by rotation
        direction: "left" or "right" for localization direction
        
    Returns:
        indices: Site pairs for Givens rotations
        givens: Givens rotation matrices (2x2)
    """
    givens_list = []
    indices_list = []
    
    M = mat.copy()
    if direction == "right":
        loc = loc[::-1]
    
    for n, k in enumerate(loc):
        if direction == "right":
            reduction = np.arange(sect[0], sect[-1] - n)
        else:
            reduction = np.arange(sect[-1] - 1, sect[0] + n - 1, -1)
        
        i_n, g_n = [], []
        v = M[:, k].copy()
        
        for j in reduction:
            q, p = v[j], v[j + 1]
            norm = np.sqrt(np.abs(p)**2 + np.abs(q)**2)
            
            if norm < 1e-14:
                g = np.eye(2, dtype=complex)
            else:
                if direction == "left":
                    g = np.array([[q.conj(), p.conj()], [-p, q]], dtype=complex) / norm
                    v[j], v[j + 1] = norm, 0
                else:
                    g = np.array([[p, -q.conj()], [q, p.conj()]], dtype=complex) / norm
                    v[j], v[j + 1] = 0, norm
            
            i_n.append([j, j + 1])
            g_n.append(g)
        
        if i_n:
            rot = RotFromGivens(i_n, g_n, sect)
            M = rot @ M
            indices_list.extend(i_n)
            givens_list.extend(g_n)
    
    return np.array(indices_list), np.array(givens_list)


def RotFromGivens(indices: List, givens: List, sect: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from Givens rotations.
    
    Args:
        indices: Site pairs for rotations
        givens: Rotation matrices
        sect: Affected sites
        
    Returns:
        Combined rotation matrix
    """
    rot = np.eye(len(sect), dtype=complex)
    
    for ind, g in zip(reversed(indices), reversed(givens)):
        rot[:, ind] = rot[:, ind] @ g
    
    return rot


def Apply_Fermionic_Op(A: str, Op, l: int, side: str = "L"):
    """
    Apply fermionic operator to MPS/MPO.
    
    Args:
        A: Operator type ("c", "c+", "n", "nn")
        Op: MPS or MPO object
        l: Site index
        side: "L" or "R" for MPO
        
    Returns:
        Modified operator
    """
    O = Op.copy()
    c = np.array([[0, 1], [0, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    if O.TN == "MPO":
        wh = 1 if side == "L" else 0
        wh2 = 1 if side == "L" else 2
        tr = [1, 0, 2, 3] if side == "L" else [1, 2, 0, 3]
        
        if A == "c":
            for k in range(l):
                B = O.MPO[f'B{k}']
                O.MPO[f'B{k}'] = np.tensordot(sz, B, axes=([wh], [wh2])).transpose(tr)
            
            B = O.MPO[f'B{l}']
            O.MPO[f'B{l}'] = np.tensordot(c, B, axes=([wh], [wh2])).transpose(tr)
            
        elif A == "n":
            n = 0.5 * (np.eye(2, dtype=complex) - sz)
            B = O.MPO[f'B{l}']
            O.MPO[f'B{l}'] = np.tensordot(n, B, axes=([wh], [wh2])).transpose(tr)
    
    elif O.TN == "MPS":
        tr = [1, 0, 2]
        
        if A in ["c", "c+"]:
            for k in range(l):
                B = O.MPS[f'B{k}']
                O.MPS[f'B{k}'] = np.tensordot(sz, B, axes=([1], [1])).transpose(tr)
            
            c_op = c if A == "c" else c.T
            B = O.MPS[f'B{l}']
            O.MPS[f'B{l}'] = np.tensordot(c_op, B, axes=([1], [1])).transpose(tr)
            
        elif A == "n":
            n = 0.5 * (np.eye(2, dtype=complex) - sz)
            B = O.MPS[f'B{l}']
            O.MPS[f'B{l}'] = np.tensordot(n, B, axes=([1], [1])).transpose(tr)
            
        elif A == "nn":
            nn = np.zeros((4, 4), dtype=complex)
            nn[0, 0] = nn[0, 3] = nn[3, 0] = nn[3, 3] = 1
            nn[0, 3] = nn[3, 0] = -1
            nn = nn.reshape(2, 2, 2, 2) / 2
            Apply_gate_MPS(O, nn, l, l + 1)
    
    return O


def Trace(A, B, conjA: bool = True) -> complex:
    """
    Compute trace of two MPS/MPO.
    
    Args:
        A, B: MPS or MPO objects
        conjA: Whether to conjugate A
        
    Returns:
        Trace value
    """
    L = A.L
    
    if A.TN == "MPO":
        if conjA:
            Pac = np.tensordot(A.MPO["B0"].conj(), B.MPO["B0"], axes=([0, 1, 2], [0, 1, 2]))
        else:
            Pac = np.tensordot(A.MPO["B0"], B.MPO["B0"], axes=([0, 2, 1], [0, 1, 2]))
        
        for i in range(1, L - 1):
            if conjA:
                Pac = np.tensordot(Pac, A.MPO[f"B{i}"].conj(), axes=([0], [0]))
                Pac = np.tensordot(Pac, B.MPO[f"B{i}"], axes=([0, 1, 2], [0, 1, 2]))
            else:
                Pac = np.tensordot(Pac, A.MPO[f"B{i}"], axes=([0], [0]))
                Pac = np.tensordot(Pac, B.MPO[f"B{i}"], axes=([0, 2, 1], [0, 1, 2]))
        
        if conjA:
            End = np.tensordot(A.MPO[f"B{L-1}"].conj(), B.MPO[f"B{L-1}"], 
                             axes=([1, 2, 3], [1, 2, 3]))
        else:
            End = np.tensordot(A.MPO[f"B{L-1}"], B.MPO[f"B{L-1}"], 
                             axes=([1, 2, 3], [1, 2, 3]))
    
    elif A.TN == "MPS":
        Pac = np.tensordot(A.MPS["B0"].conj(), B.MPS["B0"], axes=([0, 1], [0, 1]))
        
        for i in range(1, L - 1):
            Pac = np.tensordot(Pac, A.MPS[f"B{i}"].conj(), axes=([0], [0]))
            Pac = np.tensordot(Pac, B.MPS[f"B{i}"], axes=([0, 1], [0, 1]))
        
        End = np.tensordot(A.MPS[f"B{L-1}"].conj(), B.MPS[f"B{L-1}"], 
                         axes=([1, 2], [1, 2]))
    
    return np.tensordot(Pac, End, axes=([0, 1], [0, 1]))


def MPO_Correlation_Matrix(Op) -> np.ndarray:
    """
    Compute correlation matrix for MPO.
    
    Args:
        Op: MPO object
        
    Returns:
        Correlation matrix (2L x 2L)
    """
    O = Op.copy()
    L = O.L
    norm_O = np.abs(Trace(O, O, conjA=True))
    R = np.zeros((2 * L, 2 * L), dtype=complex)
    
    for i in range(2 * L):
        ind_i = i // 2
        side_i = "L" if i % 2 == 0 else "R"
        O1 = Apply_Fermionic_Op("c", O, ind_i, side_i)
        
        for j in range(i, 2 * L):
            ind_j = j // 2
            side_j = "L" if j % 2 == 0 else "R"
            O2 = Apply_Fermionic_Op("c", O, ind_j, side_j)
            R[i, j] = Trace(O1, O2, conjA=True) / norm_O
    
    R = R + R.conj().T - np.diag(R.diagonal())
    return R


def MPS_Correlation_Matrix(Psi, sites: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute correlation matrix for MPS.
    
    Args:
        Psi: MPS object
        sites: Sites to include (default: all sites)
        
    Returns:
        Correlation matrix
    """
    c = np.array([[0, 1], [0, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = Psi.L
    
    # Compute normalization
    norm = np.ones((1, 1), dtype=complex)
    for i in range(L - 1):
        norm = np.tensordot(norm, Psi.MPS[f"B{i}"].conj(), axes=([0], [0]))
        norm = np.tensordot(norm, Psi.MPS[f"B{i}"], axes=([0, 1], [0, 1]))
    norm = np.tensordot(norm, Psi.MPS[f"B{L-1}"].conj(), axes=([0], [0]))
    norm = np.abs(np.tensordot(norm, Psi.MPS[f"B{L-1}"], axes=([0, 1, 2], [0, 1, 2])))
    
    if sites is None:
        sites = np.arange(L)
    
    Q = np.zeros((len(sites), len(sites)), dtype=complex)
    
    for ind_i, i in enumerate(sites):
        Bi = Psi.MPS[f"B{i}"]
        Bic = np.tensordot(c, Bi, axes=([1], [1])).transpose(1, 0, 2)
        Lam = np.diag(Psi.MPS[f"Lam{i}"])
        
        A = np.tensordot(Lam, Bic, axes=([1], [0]))
        Q[ind_i, ind_i] = np.tensordot(A.conj(), A, axes=([0, 1, 2], [0, 1, 2]))
        
        Bc = np.tensordot(Lam, Bic, axes=([1], [0]))
        B = np.tensordot(Lam, Bi, axes=([1], [0]))
        Pac = np.tensordot(Bc.conj(), B, axes=([0, 1], [0, 1]))
        
        for ind_j, j in enumerate(sites[ind_i + 1:], start=ind_i + 1):
            Bj = Psi.MPS[f"B{j}"]
            Bjc = np.tensordot(c, Bj, axes=([1], [1])).transpose(1, 0, 2)
            
            A = np.tensordot(Pac, Bj.conj(), axes=([0], [0]))
            Q[ind_i, ind_j] = np.tensordot(A, Bjc, axes=([0, 1, 2], [0, 1, 2]))
            
            Bjz = np.tensordot(sz, Bj, axes=([0], [1])).transpose(1, 0, 2)
            Pac = np.tensordot(Pac, Bjz.conj(), axes=([0], [0]))
            Pac = np.tensordot(Pac, Bj, axes=([0, 1], [0, 1]))
    
    Q = Q / norm
    Q = Q + Q.conj().T - np.diag(Q.diagonal())
    return Q


def Apply_gate_MPS(Psi, gate: np.ndarray, l1: int, l2: int) -> None:
    """
    Apply 2-body gate to MPS (in-place).
    
    Args:
        Psi: MPS object
        gate: Gate tensor (2,2,2,2)
        l1, l2: Site indices
    """
    MPS = Psi.MPS
    lshape = MPS[f"B{l1}"].shape[0]
    rshape = MPS[f"B{l2}"].shape[-1]
    
    PhiB = np.tensordot(MPS[f"B{l1}"], MPS[f"B{l2}"], axes=([2], [0]))
    UPhiB = np.tensordot(gate, PhiB, axes=([2, 3], [1, 2]))
    UPhiB = UPhiB.transpose([2, 0, 1, 3])
    
    Phi = np.tensordot(np.diag(MPS[f"Lam{l1}"]), UPhiB, axes=([1], [0]))
    Phi = Phi.reshape(lshape * 2, rshape * 2)
    
    U, S, Vh = np.linalg.svd(Phi, full_matrices=False)
    
    mask = (S / np.linalg.norm(S)) > 1e-10
    S = S[mask] / np.linalg.norm(S[mask])
    
    B2 = Vh[mask, :].reshape(-1, 2, rshape)
    B1 = np.tensordot(UPhiB, B2.conj(), axes=([2, 3], [1, 2]))
    
    Psi.MPS[f"B{l1}"] = B1
    Psi.MPS[f"Lam{l2}"] = S
    Psi.MPS[f"B{l2}"] = B2
    Psi.BondDim[l2] = S.shape[0]


def Apply_gate_MPO(Psi, gate: np.ndarray, l1: int, l2: int, 
                   bothsides: bool = True, side: str = "L", 
                   th: float = 1e-12, right_gate: Optional[np.ndarray] = None) -> None:
    """
    Apply 2-body gate to MPO (in-place).
    
    Args:
        Psi: MPO object
        gate: Gate tensor (2,2,2,2)
        l1, l2: Site indices
        bothsides: Apply gate on both sides
        side: "L" or "R" for one-sided application
        th: SVD truncation threshold
        right_gate: Optional different gate for right side
    """
    MPO = Psi.MPO
    ldim = MPO[f"B{l1}"].shape[0]
    rdim = MPO[f"B{l2}"].shape[-1]
    
    if not bothsides and side == "R":
        ind = [2, 4]
        tr = [2, 3, 0, 4, 1, 5]
    else:
        ind = [1, 3]
        tr = [2, 0, 3, 1, 4, 5]
    
    PhiB = np.tensordot(MPO[f"B{l1}"], MPO[f"B{l2}"], axes=([3], [0]))
    UPhiB = np.tensordot(gate, PhiB, axes=([2, 3], ind)).transpose(tr)
    
    if bothsides:
        UPhiB = np.tensordot(gate.conj(), UPhiB, axes=([2, 3], [2, 4]))
        UPhiB = UPhiB.transpose([2, 3, 0, 4, 1, 5])
    
    if right_gate is not None:
        UPhiB = np.tensordot(right_gate, UPhiB, axes=([2, 3], [2, 4]))
        UPhiB = UPhiB.transpose([2, 3, 0, 4, 1, 5])
    
    Phi = np.tensordot(np.diag(MPO[f"Lam{l1}"]), UPhiB, axes=([1], [0]))
    Phi = Phi.reshape(ldim * 2 * 2, rdim * 2 * 2)
    
    U, S, Vh = np.linalg.svd(Phi, full_matrices=False)
    
    mask = (S / np.linalg.norm(S)) > th
    S = S[mask] / np.linalg.norm(S[mask])
    
    B2 = Vh[mask, :].reshape(-1, 2, 2, rdim)
    B1 = np.tensordot(UPhiB, B2.conj(), axes=([3, 4, 5], [1, 2, 3]))
    
    Psi.MPO[f"B{l1}"] = B1
    Psi.MPO[f"Lam{l2}"] = S
    Psi.MPO[f"B{l2}"] = B2
    Psi.BondDim[l2] = S.shape[0]


def apply_MPO_MPS(O, Psi):
    """Apply MPO to MPS."""
    Res = classMPS(Psi.L)
    
    Res.MPS["B0"] = np.tensordot(O.MPO["B0"], Psi.MPS["B0"], axes=([2], [1]))
    Res.MPS["B0"] = Res.MPS["B0"].transpose([0, 3, 1, 2, 4]).reshape(1, 2, -1)
    Res.MPS["Lam0"] = np.ones(1, dtype=complex)
    
    for i in range(1, Psi.L):
        A = np.tensordot(Res.MPS[f"B{i-1}"], O.MPO[f"B{i}"], axes=([2], [0]))
        A = np.tensordot(A, Psi.MPS[f"B{i}"], axes=([2, 3], [0, 1]))
        
        ldim = A.shape[0]
        rdim = A.shape[3] * A.shape[4]
        
        B = np.tensordot(np.diag(Res.MPS[f"Lam{i-1}"]), A, axes=([1], [0]))
        B[np.abs(B) < 1e-20] = 0
        
        _, S, V = np.linalg.svd(B.reshape(ldim * 2, 2 * rdim), full_matrices=False)
        
        mask = S / np.linalg.norm(S) > 1e-10
        S = S[mask] / np.linalg.norm(S[mask])
        
        B2 = V[mask, :].reshape(-1, 2, O.MPO[f"B{i}"].shape[3], Psi.MPS[f"B{i}"].shape[2])
        B1 = np.tensordot(A, B2.conj(), axes=([2, 3, 4], [1, 2, 3]))
        
        Res.MPS[f"B{i-1}"] = B1
        Res.MPS[f"Lam{i}"] = S
        Res.MPS[f"B{i}"] = B2
        Res.BondDim[i] = S.shape[0]
    
    Res.MPS[f"B{Psi.L-1}"] = Res.MPS[f"B{Psi.L-1}"].reshape(
        Res.MPS[f"B{Psi.L-1}"].shape[0], 2, 1
    )
    
    return Res


# ==================== MAIN SCRIPT ====================

if __name__ == "__main__":
    t0 = time.time()
    
    # System parameters
    L = 16
    d = 2
    
    model = "IRLM"
    save = False
    
    # TN parameters
    phys_dims = 1
    alpha = -1
    truncation_type = "global"
    chi_max = 1024
    s = L // 2 if alpha == L + 1 else L
    trunc_sing_vals = 1e-10
    data_as_tensors = False
    
    # Time evolution parameters
    TrottOrder = 4
    dt = DT = 0.01
    T = 1.1
    nStepsCalc = 100
    
    # Storage arrays
    Nsteps = int(T / dt)
    nComp = 10 # Nsteps // nStepsCalc + (Nsteps % nStepsCalc > 0)
    Rev_t      = np.zeros((2 * L, nComp), dtype=float)
    CumErr_t   = np.zeros((L, nComp), dtype=float)
    AbsTime    = np.zeros(nComp, dtype=float)
    Norm       = np.zeros(nComp, dtype=float)
    BondDim    = np.zeros((nComp, L + 1), dtype=int)
    BondDimRot = np.zeros((nComp, L + 1), dtype=int)
    Time       = np.zeros(nComp)
    
    # Model parameters
    params = {
        'L': L, 'dt': dt, 'alpha': alpha, 'Truncation': truncation_type,
        'chi': chi_max, 'Sect': s, 'TrottOrder': TrottOrder, 'Nsteps': Nsteps,
        'phys_dims': phys_dims
    }
    
    if model == 'Heis_nn':
        J = 1
        Jz = 0
        params.update({
            'd': d, 'J': J, 'Jz': Jz, 'Uint': 0, 'V': 0,
            'gamma': 0, 'periodT': 0, 'hmean': 0, 'hdrive': 0
        })
        seq0 = [np.array((i, i + 1)) for i in range(0, L - 1, 2)]
        seq1 = [np.array((i, i + 1)) for i in range(1, L - 1, 2)]
        gates_layers = {"H0": seq0, "H1": seq1}
        
    elif model == 'IRLM':
        Uint = 0.2
        V = 0.2
        gamma = 0.5
        ed = 0
        params.update({
            'L': L, 'd': d, 'Uint': Uint, 'V': V,
            'gamma': gamma, 'ed': ed, 'J': 0, 'Jz': 0
        })
        seq0 = [np.array((0, 1))]
        seq1 = [np.array((i, i + 1)) for i in range(2, L - 1, 2)]
        seq2 = [np.array((i, i + 1)) for i in range(1, L - 1, 2)]
        gates_layers = {"H0": seq0, "H1": seq1, "H2": seq2}
    
    # Print parameters
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")
    
    # Trotter decomposition
    nParts = len(gates_layers)
    TrotterSteps, U, _, _ = trott.Trotter_Seq(
        TrottOrder, Nsteps, dt, nParts, nStepsCalc, params,
        alpha=alpha, data_as_tensors=data_as_tensors, model=model,
        Hsteps=list(gates_layers.keys()), symm=False
    )
    
    # Initialize file for saving
    filename = f"../data/MPO/{model}/U_{model}"
    for key, value in params.items():
        filename += f"_{key}{value}"
    
    if save:
        f = h5.File(filename + ".h5", "w")
    
    # Initialize operator
    Op = classMPO(L, "Id")
    
    # Time evolution with TEBD
    t, cpt, step_t0 = 0.0, 0, 0
    st = time.time()
    
    for step, dt_step, newTstep, ComputeObs in TrotterSteps[step_t0:]:
        # Apply gates for the given layer
        for (l1, l2) in gates_layers[step]:
            Apply_gate_MPO(Op, U[step, dt_step], l1, l2, bothsides=False)
        
        # Real time step done
        if newTstep:
            t += DT
            print(Op.BondDim)
            print(f"Evolution time: {np.round(t, 4)}, "
                  f"Computational time: {time.time() - st}\n")
            st = time.time()
            
            if np.abs(t - 0.5) < 1e-2 or np.abs(t - 1.0) < 1e-2 or np.abs(t - 1.5) < 1e-2:
                Op.compress(1e-20)
                BondDim[cpt] = Op.BondDim
                
                R = MPO_Correlation_Matrix(Op)
                
                Oprot = Op.copy()
                Oprot.rotate_Unit(R)
                Oprot.compress(1e-14)
                Oprot.EE()
                
                BondDimRot[cpt] = Oprot.BondDim
                Time[cpt] = t
                cpt += 1
    
    print("Temps final:", time.time()-t0)
    print(stop)
    
    # Save results
    filename = "C:/Users/maxim/Documents/GitHub/Few_body_operators/Codes/IRLM_U"
    for key, value in params.items():
        filename += f"_{key}{value}"
    
    with h5.File(filename + ".h5", "w") as f:
        f.create_dataset("BondDim", data=BondDim)
        f.create_dataset("BondDimRot", data=BondDimRot)
        f.create_dataset("t", data=Time)
    
    print("Simulation completed successfully!")
    
    # Post-processing: Rotation analysis
    R = MPO_Correlation_Matrix(Op)
    Rev, Revecs = np.linalg.eigh(R)
    
    A = R[::2, 1::2].copy()
    Aev, Aevecs = np.linalg.eigh(A)
    
    U_svd, S, Vd = np.linalg.svd(A)
    
    print("A eigenvalues: ", 0.5 - np.abs(Aev))
    print("R eigenvalues: ", Rev)
    
    # Trace with MPO
    Uni = Apply_Fermionic_Op("n", Op, 0, "R")
    niU = Apply_Fermionic_Op("n", Op, 0, "L")
    Sz_ex = Trace(niU, Uni)
    print(f"\nTr(U+(t) ni U(t) ni) = {Sz_ex}")
    
    # Occupation analysis
    occ = Rev > 0.5
    D = Revecs[:, occ]
    
    Sz = (np.sum(D[0, :].conj() * D[0, :]) - 
          np.sum(D[0, :, None].conj() * D[0, :, None] * 
                 D[1, None, :].conj() * D[1, None, :]) + 
          np.sum(D[0, :, None].conj() * D[1, :, None] * 
                 D[1, None, :].conj() * D[0, None, :]))
    print(f"Err rot Wick: {1 - Sz / Sz_ex}")
    
    print("Before rotation:")
    print(Op.BondDim)
    
    # Rotate orbitals
    Op.rotate_Unit(R)
    Op.compress(1e-13)
    Op.EE()
    
    print("After rotation:")
    print(Op.BondDim)
    
    print("\nDiagonal of real R:", np.abs(R.diagonal()))