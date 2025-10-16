import numpy as np
import scipy as sp
from scipy.sparse import kron
import h5py as h5

import matplotlib as mpl 
import matplotlib.pyplot as plt

## Luca's libraries for information lattice
#from tqdm import tqdm
#sys.path.append("../../information_lattice/src/")
#import information as il
#import tensors as tn
#from misc import colors
#sys.path.append("../../information_lattice/plotting/plot_gen/")
#import plot_information as pltil


## ---------------------------------------------------------
class Fermionic_Operators():
	def __init__(self, L):
		sp_i  = np.array(((0,0), (1,0)) )  ## can be sigma+
		sm_i = np.array( ((0,1), (0,0)) )  ## can be sigma-
		sz_i  = np.array(((1,0), (0,-1)) ) ## can be sz

		## We define the occuation super-operators in the many-body basis
		self.sp  = np.zeros(L, dtype=object)
		self.sm = np.zeros(L, dtype=object)
		self.sz = np.zeros(L, dtype=object)

		self.c = np.zeros(L, dtype=object)
		self.n = np.zeros(L, dtype=object)

		Pstring = np.identity(2**L)
		for i in range(L):
			sp_i_L = kron(np.identity(2**i), np.kron(sp_i, np.identity(2**(L-i-1))))
			sm_i_L = np.kron(np.identity(2**i), np.kron(sm_i, np.identity(2**(L-i-1))))
			sz_i_L = np.kron(np.identity(2**i), np.kron(sz_i, np.identity(2**(L-i-1))))

			self.sp[i] = sp_i_L
			self.sm[i] = sm_i_L
			self.sz[i] = sz_i_L

			self.c[i]  = (Pstring @ sm_i_L).astype(complex)
			self.n[i]  = 0.5 * (np.identity(2**L) - sz_i_L)

			Pstring = Pstring @ sz_i_L

		## Operators for the 1-RDM: (S stands for super-operator)
		self.S_didj = np.zeros((2*L,2*L), dtype=object)

		for i in range(2*L):
			#S_di = kron(self.sm[i], sp.sparse.identity(2**L) if (i < L) else kron(sp.sparse.identity(2**L), self.sm[i-L].T)
			S_di = kron(self.c[i], sp.sparse.eye(2**L)) if (i < L) else kron(sp.sparse.eye(2**L), self.c[i-L].T)
			for j in range(2*L):
				#S_dj = kron(self.sm[j], sp.sparse.identity(2**L)) if (j < L) else kron(sp.sparse.identity(2**L), self.sm[j-L].T)
				S_dj = kron(self.c[j], sp.sparse.eye(2**L)) if (j < L) else kron(sp.sparse.eye(2**L), self.c[j-L].T)

				self.S_didj[i,j] = S_di.conj().T @ S_dj 

## ---------------------------------------------------------
class Vectorized_Operator():
	def __init__(self, O, L, order="C"):
		self.size = 2*int(L)
		self.L = int(L)
		self.vec = O.ravel(order=order)			## Vectorize it
		self.vec = self.vec / np.linalg.norm(self.vec)

	def __repr__(self):
		return self.vec

	def show_dec(self):
		wh = np.where(self.vec)
		for i in wh[0]:
			if np.abs(self.vec[i]) > 1e-14:
				print(np.binary_repr(i, width=self.size), self.vec[i])
		print("Norm: ", np.linalg.norm(self.vec))
        

	## Rotates the vectorized operator
	def one_body_rot(self, rot, Ops):
		MB_rot = np.zeros((4**self.L, 4**self.L), dtype=complex)
		Gamma = sp.linalg.logm(rot)
		for i in range(2*self.L):
			for j in range(2*self.L):
				MB_rot += Gamma[i,j] * Ops.S_didj[i,j]
		MB_rot = sp.linalg.expm(-MB_rot)
		vecO_rotate = Vectorized_Operator(np.identity(2**self.L), self.L)
		vecO_rotate.vec = MB_rot @ self.vec
		
		return vecO_rotate
	
	## Computes the EE of the state
	def EE(self):
		tens = np.reshape(self.vec, (2,)*2*self.L)
		tens = np.moveaxis(tens, np.arange(self.L, 2*self.L), np.arange(1,2*self.L,2)) ## put ~dual and direct spaces next to each others
		for i in range(0,2*self.L+1):
			tens_bipartite = np.reshape(tens, (2**(i), 2**(2*self.L-i)))
			uu, ss, vv = np.linalg.svd(tens_bipartite, full_matrices=False)
			norm = np.sqrt(np.sum(ss**2))
			mask = (ss/norm) > 1e-8
			ss = ss[mask] / norm
			print(i, "Bond dim:", ss.shape, "S =",np.sum(-np.log(ss**2)*ss**2))#, ss)

## ---------------------------------------------------------
def  Givens_rotations(mat, start = 0, side = "right"):
    M = mat.copy()
    Lx,Ly = M.shape[0], M.shape[1]
    givens = np.zeros((Ly,Lx,2,2), dtype=complex)
    indices = []
    if side == "left":
        for i in range(Ly):
            for j in range(np.min((Lx-1,Lx-1-i))):
                p, q = M[-2-j,i], M[-1-j,i]
                theta = np.arctan(q/p) if np.abs(p) > 1e-15 else 0
                c, s = np.cos(theta), np.sin(theta)
                g = np.array(((c,s),(-np.conj(s),c)))
                givens[i,j,:,:] = g
                
                M[Lx-2-j:Lx-j,i] = g @ M[Lx-2-j:Lx-j,i] ## eliminate component Lx-2-j
                indices.append((Lx-2-j,Lx-1-j))
    elif side == "right": ## Does not work
        for i in range(Ly-1):
            for j in range(np.min((Lx-1,Lx-1-i))):
                p, q = M[j,i], M[j+1,i]
                theta = np.arctan(-p/q).conj() if np.abs(q) > 1e-15 else 0
                c, s = np.cos(theta), np.sin(theta)
                g = np.array(((c,s),(-np.conj(s),c)))
                givens[i,j,:,:] = g
                
                M[j:j+2,i] = g @ M[j:j+2,i] ## eliminate component Lx-2-j
                indices.append((j,j+1))
            M[:,i+1] = M[:,i+1] - M[:,i+1] * M[:,i].conj().T
            M[:,i+1] =  M[:,i+1] / np.linalg.norm(M[:,i+1])
    indices = np.array(indices)
    ## start corresponds to the first physical index from which the matrix acts on.
    indices += start
    print(M)
    return indices, givens

## ---------------------------------------------------------
def Gates_from_Givens(givens):
        Lx, Ly = givens.shape[0], givens.shape[1]
        gates = np.zeros((np.count_nonzero(givens, axis=(0,1))[0,0],2,2,2,2), dtype=complex)
        
        g = np.zeros((4,4), dtype=complex)
        g[0,0] = 1
        g[3,3] = 1
        c = 0
        for i in range(Ly):
            for j in range(np.min((Lx-1,Lx-1-i))):
                g[1:3,1:3] = givens[i,j,:,:].conj().T
                gates[c] = g.reshape((2,)*4)
                c += 1
        return gates

## ---------------------------------------------------------
def OpEE(O):
	O = np.reshape(O.copy(), (2,)*2*L)
	for i in range(0,L+1):
		a = np.concatenate((np.arange(0,i), np.arange(L,L+i)))
		b = np.concatenate((np.arange(i,L), np.arange(L+i,2*L)))
		source = np.concatenate((a,b))
		destin = np.arange(2*L)
		O_bipartite = np.moveaxis(O, source, destin).reshape(2**(2*i), 2**(2*(L-i)))
		uu, ss, vv = np.linalg.svd(O_bipartite, full_matrices=False)
		norm = np.sqrt(np.sum(ss**2))
		mask = (ss/norm) > 1e-8
		ss = ss[mask] / norm
		print(i, "Bond dim:", ss.shape, "S =",np.sum(-np.log(ss**2)*ss**2))#, ss)
	print('Norm: ', norm)

## ---------------------------------------------------------
def RDM1(vec, super_op):
	RDM1 = np.zeros((2*L,2*L), dtype=complex)
	for i in range(2*L):
		for j in range(2*L):
			RDM1[i,j] = (vec.conj().T @ super_op[i,j] @ vec)
	return RDM1

## ---------------------------------------------------------
def Diagonal(n_MB, L, j=2, dt=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	ek = np.arange(-int(L/2), int(L/2))
	for k in range(L):
		H += ek[k] * n_MB[k]

	U = sp.linalg.expm(-1j * H * dt)
	Sz_j_t = U.conj().T @ (n_MB[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t

## ---------------------------------------------------------
def SYK(c_MB, n_MB, L, j=2, dt=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	t = np.random.random((L,L,L,L))
	for i in range(L):
		for j in range(L):
			for k in range(L):
				for l in range(L):
					H += t[i,j,k,l] * (c_MB[i].T @ c_MB[j].T @ c_MB[k] @ c_MB[l] + c_MB[l].T @ c_MB[k].T @ c_MB[j] @ c_MB[i])

	U = sp.linalg.expm(-1j * H * dt)
	Sz_j_t = U.conj().T @ (n_MB[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t
	
## ---------------------------------------------------------
def Heisenberg(c_MB, n_MB, L, J, Jz, j=2, dt=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	for i in range(L-1):
		H += J/2 * (c_MB[i].conj().T @ c_MB[i+1] + (c_MB[i].conj().T @ c_MB[i+1]).conj().T) + Jz * n_MB[i] @ n_MB[i+1]
		#H += J/2 * (c_MB[i].conj().T @ c_MB[i+1] + (c_MB[i].conj().T @ c_MB[i+1]).conj().T) + Jz * (n_MB[i]-1/2*sp.sparse.identity(2**L)) @ (n_MB[i+1]-1/2*sp.sparse.identity(2**L))

	U = sp.linalg.expm(-1j * H * dt)
	Sz_j_t = U.conj().T @ (n_MB[j]) @ U

	return H, U, Sz_j_t

## ---------------------------------------------------------
def MBL(c_MB, n_MB, L, J, Jz, j=2, dt=0.1, W=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	for i in range(L-1):
		W_i = np.random.uniform(-W,W,1)[0]
		H += J/2 * (c_MB[i].conj().T @ c_MB[i+1] + (c_MB[i].conj().T @ c_MB[i+1]).conj().T) + Jz * (n_MB[i]-1/2*sp.sparse.identity(2**L)) @ (n_MB[i+1]-1/2*sp.sparse.identity(2**L))
		H += W_i * (n_MB[i]-1/2*sp.sparse.identity(2**L)) 

	U = sp.linalg.expm(-1j * H * dt)
	Sz_j_t = U.conj().T @ (n_MB[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t

## ---------------------------------------------------------
def nnn_Heisenberg(c, n, L, J, J2, Jz, Jz2, j=2, dt=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	for i in range(L-1):
		H += J/2 * (c[i].conj().T @ c[i+1] + (c[i].conj().T @ c[i+1]).conj().T) + Jz * (n[i]-1/2*sp.sparse.identity(2**L)) @ (n[i+1]-1/2*sp.sparse.identity(2**L))
	for i in range(L-2):
		H += J2/2 * (c[i].conj().T @ c[i+2] + (c[i].conj().T @ c[i+2]).conj().T) + Jz2 * (n[i]-1/2*sp.sparse.identity(2**L)) @ (n[i+2]-1/2*sp.sparse.identity(2**L))

	U = sp.linalg.expm(-1j * H * dt)
	Sz_j_t = U.conj().T @ (n[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t
	
## ---------------------------------------------------------
def XX(c, n, L, j=2, dt=0.1):
	H = np.zeros((2**L,2**L), dtype=complex)
	for i in range(L-1):
		H += (c[i].conj().T @ c[i+1] + c[i+1].conj().T @ c[i]) 
    
	U = sp.linalg.expm(-1j * H * dt)
	print(dt)

	Sz_j_t = U.conj().T @ (n[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t

## ---------------------------------------------------------
def IRLM(c, n, L, U=0, V=0.1, t=0.5, j=0, dt=0.1):
	print("IRLM", U, V, t)
	H = np.zeros((2**L,2**L), dtype=complex)
	#H += V * (Ops.sp[0] @ Ops.sm[1] + Ops.sm[0] @ Ops.sp[1]) + U * Ops.sz[0] @ Ops.sz[1] * 1/4
	H += V * (c[0].conj().T @ c[1] + c[1].conj().T @ c[0]) + U * (n[0]-1/2*sp.sparse.identity(2**L)) @ (n[1]-1/2*sp.sparse.identity(2**L))
	for i in range(1,L-1):
		H += t * (c[i].conj().T @ c[i+1] + c[i+1].conj().T @ c[i])

	U = sp.linalg.expm(-1j * H * dt)
	#Sz_j_t = U.conj().T @ (n_MB[j]) @ U
	Sz_j_t = U.conj().T @ (n[j] - 1/2 * np.identity(2**L)) @ U

	return H, U, Sz_j_t

## ---------------------------------------------------------
def Spectrum_R_Op(Op, Ops, v=False):
	## We vectorize it
	vecO = Vectorized_Operator(Op, L)
	## Defines the 1-RDM
	rho = RDM1(vecO.vec, Ops.S_didj)
	print(np.round(rho,4)) if v else None

	## Diagonalize it
	ev, evecs = np.linalg.eigh(rho)

	print("Rev: \n", ev) if v else None
	
	return ev, evecs

## ---------------------------------------------------------
def Spectrum_Q_St(St, Ops, v=False):
	St = St.reshape(-1,1)
	Q = np.zeros((L,L), dtype=complex)
	for i in range(L):
		for j in range(L):
			Q[i,j] = (St.conj().T @ Ops.c[i].conj().T @ Ops.c[j] @ St)[0,0]
	ev, evecs = np.linalg.eigh(Q)

	print("Qev: \n", ev) if v else None
	
	return ev, evecs

## ---------------------------------------------------------
def ChooseModel(model):
	## Choose the model
	if model == "Heisenberg":
		params = {"J":J, "Jz":Jz, "L":L, "dt":dt}
		H, U, Sz_j_t = Heisenberg(Ops.c, Ops.n, j=int(L/2), **params)
	elif model == "XX":
		params = {"L":L, "dt":dt}
		H, U, Sz_j_t = XX(Ops.c, Ops.n, j=int(L/2),  **params)
	elif model == "IRLM":
		params = {"U":Uint, "V":0.2, "t":0.5, "L":L, "dt":dt}
		H, U, Sz_j_t = IRLM(Ops.c, Ops.n, j=0, **params)
	elif model == "nnn_Heisenberg":
		params = {"J":1, "J2":1, "Jz":1, "Jz2":1, "L":L, "dt":dt}
		H, U, Sz_j_t = nnn_Heisenberg(Ops.c, Ops.n, j=int(L/2), **params)
	elif model == "Diagonal":
		params = {"L":L, "dt":dt}
		H, U, Sz_j_t = Diagonal(Ops.n, j=int(L/2), **params)
	elif model == "SYK":
		params = {"L":L, "dt":0.1}
		H, U, Sz_j_t = SYK(Ops.c, Ops.n, j=int(L/2), **params)
	if model == "MBL":
		params = {"J":1, "Jz":1, "L":L, "dt":4, "W":10.0}
		H, U, Sz_j_t = MBL(Ops.c, Ops.n, j=int(L/2), **params)
	
	return H, U, Sz_j_t, params

## ---------------------------------------------------------
def Information_Lattice(psi, plot = False, title=""):
	psi_MPS = tn.MPS(psi, L=2*L)
	entropies = il.get_entropy_infolattice(psi_MPS)
	local_inf = il.entropy_to_local_information(entropies)

	maxval = 0
	minval = 0
	for key in local_inf.keys():
		maxval = max(maxval, np.max(local_inf[key]))
		minval = min(minval, np.min(local_inf[key]))

	if plot == True:
		norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
		fig, ax = plt.subplots()
		pltil.plot_lattice(ax, local_inf, 2*L, norm)
		ax.set_title(title)
		col = colors("nord")
		cmap = col.get_cmap()
		cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

		plt.savefig("Information_%s.pdf"%(model), dpi=300)

	return local_inf

## ---------------------------------------------------------
## ---------------------------------------------------------

L     = 6
model = "Heisenberg" # "Diagonal", "SYK", "Heisenberg", "XX", "IRLM", "nnn_Heisenberg", "MBL"
alpha = -1
T, dt = 8, 0.5

save  = False

## Define the fermionic c,c+,n,d operators
Ops  = Fermionic_Operators(L)

## Only for MBL
if model == "MBL":
	n_real = 500
	Rspectrum = np.zeros((3,2*L))
	gap = np.zeros((8))
	t_list = np.arange(0, 4, 0.5)
	for i in tqdm(range(n_real)):
		H, U, Sz_j_t, params = ChooseModel(model)
		for j, t in enumerate(t_list):
			Ut = np.linalg.matrix_power(U.copy(), int(t/dt))
			Op = Ut
			Rev, Revecs = Spectrum_R_Op(Op, Ops, v=False)
			gap[j] += (Rev[L]-Rev[L-1]) / n_real
			#Rspectrum[j] += Rev / n_real
	plt.plot(t_list, gap, marker='o', linestyle='None')
	#for i in range(3):
	#	plt.plot(np.arange(2*L), Rspectrum[i], marker='o', linestyle='None')
	plt.show()

else:	
	J, Jz = 1, 1.0
	Uint = 0.01
	H, U, Sz_j_t, params = ChooseModel(model)

## Open the h5 file
if save:
	filename = "../data/ED/%s/R_spectrum_%s"%(model,model)
	for key in params.keys():
		filename = filename + str("_"+key+str(params[key]))
	f =  h5.File(filename+".h5", "w")

## ---------------------------------------------------------
## ---------------------------------------------------------
## Test rotations givens
# Hev, Hevecs = np.linalg.eigh(H)
# Op = Hevecs[:,0]
     
# print("\nState without rotation\n")
# vec = Vectorized_Operator(Op, int(L/2))
# vec.EE()

# rho = np.zeros((L-2,L-2), dtype=complex)
# for i in range(2,L):
# 	for j in range(2,L):
# 		rho[i-2,j-2] = (vec.vec.conj().T @ Ops.c[i].T @ Ops.c[j] @ vec.vec)
# Qev, Qevecs = np.linalg.eigh(rho)

# order = np.argsort(Qev*(1-Qev))[::-1]
# Qev = Qev[order]
# Qevecs = Qevecs[:,order]
# print(Qev)
# Rot = np.eye(L, dtype=complex)
# Rot[2:,2:] = Qevecs

# indices, givens = Givens_rotations(Rot[2:,2:], start=2)
# gates = Gates_from_Givens(givens)
# MPS_Op = Op.reshape((2,)*L)
# for i in range(indices.shape[0]):
#     # print(indices[i], gates[i].reshape(4,4))
#     MPS_Op = np.tensordot(MPS_Op, gates[i], axes=(indices[i],[2,3]))
    
# MPS_Op = MPS_Op.reshape(2**L)
# vec2 = Vectorized_Operator(Op, int(L/2))
# vec2.vec = MPS_Op
# print("\nState with givens rotation\n")
# vec2.EE()

# MB_rot = np.zeros((2**L, 2**L), dtype=complex)
# Gamma = sp.linalg.logm(Rot.conj())
# for i in range(L):
# 	for j in range(L):
# 		MB_rot += Gamma[i,j] * Ops.c[i].T @ Ops.c[j]
# MB_rot = sp.linalg.expm(-MB_rot)

# vecO_rotate = Vectorized_Operator(np.identity(2**(int(L/2))), int(L/2))
# vecO_rotate.vec = MB_rot @ Op
# print("\nState with normal rotation\n")
# vecO_rotate.EE()

# T_gate = np.kron(np.array(((1,0),(0,np.exp(1j*np.pi/4)))), np.identity(2**(L-1)))
# Op = U@U
# Op = T_gate @ Op

# ## Diagonalization of R and A
# vecO = Vectorized_Operator(Op, L)
# rho = RDM1(vecO.vec, Ops.S_didj)
# Rev, Revecs = np.linalg.eigh(rho)
# print("Spectrum: ", Rev)

# print(stop)

# A = rho[L:,:L]
# AA = A.conj().T @ A
# Aev, Aevecs = np.linalg.eigh(AA)

# # We have to sort in decreasing activity
# order = np.argsort(Rev*(1-Rev))[::-1]
# Rev = Rev[order]
# Revecs = Revecs[:,order]

# order = np.argsort(np.abs(0.25-Aev))[::-1]
# Aev = np.abs(0.25-Aev[order])
# Aevecs = Aevecs[:,order]

# Aevecs2 = Revecs * np.sqrt(2)

# print("Ev. A: ", Aev)
# print("Ev. R: ", Rev)

# ## single-body rotation of the operator
# MB_rot1 = np.zeros((2**L, 2**L), dtype=complex)
# MB_rot2 = np.zeros((2**L, 2**L), dtype=complex)
# Gamma1 = sp.linalg.logm(Aevecs.conj())
# Gamma2 = sp.linalg.logm(-Aevecs.conj())
# for i in range(L):
#         for j in range(L):
#                 MB_rot1 += Gamma1[i,j] * Ops.c[i].T @ Ops.c[j]
#                 MB_rot2 += Gamma2[i,j] * Ops.c[i].T @ Ops.c[j]
# rot1 = sp.linalg.expm(- MB_rot1)
# rot2 = sp.linalg.expm(- MB_rot2)

# print("\n Original operator \n")
# OpEE(Op)

# # OpEE of the rotated operator
# print("\n Rotated operator \n")
# Op_rot = rot1 @ Op @ rot2.T
# OpEE(Op_rot)

# print("\n Super-Rotated operator \n")
# vecO_rotate = vecO.one_body_rot(Revecs.conj(), Ops)
# vecO_rotate.EE()

# print("\n Super-Rotated operator reshaped as an MPO \n")
# A2 = vecO_rotate.vec.reshape(2**L,2**L)
# OpEE(A2)

# Op = Sz_j_t
# S_list = []
# t = dt
# t_list = []
# for tt in range(10):
# 	vecO = Vectorized_Operator(Op, L)
# 	## Defines the 1-RDM
# 	rho = RDM1(vecO.vec, Ops.S_didj)
# 	rev, revecs = np.linalg.eigh(rho)
# 	print("rho: \n",rev)

# 	A = rho[L:,:L]
# 	print(A)
# 	#ev, evecs = np.linalg.eigh(A)
# 	ev, evecs = np.linalg.eig(A[:2,:2])

# 	A0 = Ops.n[int(L/2)]
# 	A1 = H @ A0 - A0 @ H
# 	A2 = H @ A1 - A1 @ H
# 	A3 = H @ A2 - A2 @ H
# 	A4 = H @ A3 - A3 @ H
# 	A5 = H @ A4 - A4 @ H
# 	A6 = H @ A5 - A5 @ H

# 	n0 = A0 + 1j*t * A1 - (t**2/2) * A2 - 1j*(t**3/6) * A3 + (t**4/24) * A4 #+ 1j*(t**5/120) * A5 - (t**6/720) * A6

# 	norm = np.trace(A0 @ A0.conj().T)

# 	## for IRLM
# 	#A = np.zeros((2,2), dtype=complex)
# 	#A[0,0] = np.trace(n0 @ Ops.c[0].conj().T @ n0 @ Ops.c[0]) / norm
# 	#A[1,0] = np.trace(n0 @ Ops.c[1].conj().T @ n0 @ Ops.c[0]) / norm
# 	#A[0,1] = A[1,0].conj().T 
# 	#A[1,1] = np.trace(n0 @ Ops.c[1].conj().T @ n0 @ Ops.c[1]) / norm


# 	## for Heisenberg
# 	A = np.zeros((3,3), dtype=complex)
# 	A[0,0] = np.trace(n0 @ Ops.c[int(L/2)-1].conj().T @ n0 @ Ops.c[int(L/2)-1]) / norm
# 	A[1,0] = np.trace(n0 @ Ops.c[int(L/2)].conj().T   @ n0 @ Ops.c[int(L/2)-1]) / norm
# 	A[2,0] = np.trace(n0 @ Ops.c[int(L/2)+1].conj().T @ n0 @ Ops.c[int(L/2)-1]) / norm
# 	A[0,1] = np.trace(n0 @ Ops.c[int(L/2)-1].conj().T @ n0 @ Ops.c[int(L/2)])   / norm
# 	A[1,1] = np.trace(n0 @ Ops.c[int(L/2)].conj().T   @ n0 @ Ops.c[int(L/2)])   / norm
# 	A[2,1] = np.trace(n0 @ Ops.c[int(L/2)+1].conj().T @ n0 @ Ops.c[int(L/2)])   / norm
# 	A[0,2] = np.trace(n0 @ Ops.c[int(L/2)-1].conj().T @ n0 @ Ops.c[int(L/2)+1]) / norm
# 	A[1,2] = np.trace(n0 @ Ops.c[int(L/2)].conj().T   @ n0 @ Ops.c[int(L/2)+1]) / norm
# 	A[2,2] = np.trace(n0 @ Ops.c[int(L/2)+1].conj().T @ n0 @ Ops.c[int(L/2)+1]) / norm

# 	print(t, A, np.trace(n0@n0))
# 	
# 	ev, evecs = np.linalg.eig(A)

# 	print("A: ", A[1,1])
# 	ev = A[1,1]
# 	#ev = 0.5-ev[1:]

# 	#rev = np.concatenate((ev, 1-ev))
# 	rev = 1-ev 

# 	#print(np.sort(rev))

# 	t_list.append(t)
# 	S_list.append(np.sum(-np.log(np.abs(rev+1e-15)**2) * np.abs(rev+1e-15)**2))

# 	Op = U.conj().T @ Op @ U

# 	t += dt
# 	print(t_list[-1], S_list[-1])

# 	print("\n")

# np.savetxt("S_list.txt", np.stack((t_list, S_list), axis=1))

# print(kjhsdf)

# #print(Op@Op.conj().T)

# #print(np.sum(Op - Op.T))
# #
# #beta = 3
# #print(np.exp(-beta * Hev) / np.sum(np.exp(-beta * Hev)))
# #rho_beta = np.zeros((2**L,2**L), dtype=complex)	
# #for i in range(2**L):
# #	rho_beta += np.exp(-beta * Hev[i]) * np.outer(Hevecs[:,i], Hevecs[:,i])
# #Op = rho_beta / np.trace(rho_beta)

# #Op = np.outer(Hevecs[:,0], Hevecs[:,0])

# #Op = (1/2 * np.identity(2**L) - Ops.n[-1]) + (1/2 * np.identity(2**L) - Ops.n[0])

# vecO = Vectorized_Operator(Op, L)
# ## Defines the 1-RDM
# rho = RDM1(vecO.vec, Ops.S_didj)
# rev, revecs = np.linalg.eigh(rho)
# print("Rev: ", rev)
# rho[:L,:L] = 0
# rho[L:,L:] = 0
# rev, revecs = np.linalg.eigh(rho)
# print("A: \n")
# A = rho[L:,:L]
# print(A)
# #print(rho)
# print("A_ii: ", A[1,1])
# print("AA+: \n")
# print(A@A.conj().T)

# print(rev+0.5)
# #A = np.zeros((L,L), dtype=complex)
# #nrows, ncols = L,L
# #for k in [-1, 0, 1]:
# #    A += np.diag(np.diag(rho[L:,:L], k=k), k=k)
# #rho[L:,:L] = A
# #rho[:L,L:] = A.conj().T
# #rev, revecs = np.linalg.eigh(rho)
# #print(rho)
# #print(rev+0.5)
# u,s,v = np.linalg.svd(A)
# print(-s+0.5)
# for k in range(L):
# 	sig_k = 0.5*np.sqrt((1-dt**4) / (1 - 2*dt**2*np.cos(np.pi*k / (L+1)) + dt**4))
# 	print(k,-sig_k+0.5)

# print("TEST")
# A = np.zeros((L,L), dtype=complex)
# for i in range(L):
# 	for j in range(L):
# 		A[i,j] = (1j*dt)**np.abs(i-j) / sp.special.factorial(np.abs(i-j))
# 	A[i,i] -= 1/2 + 1j*dt

# u,s,v = np.linalg.svd(A)
# print(-s+0.5)
# for k in range(L):
# 	sig_k = 0.5*np.sqrt((1-dt**4) / (1 - 2*dt**2*np.cos(np.pi*k / (L+1)) + dt**4))
# 	print(k,-sig_k+0.5)
# rho[:L,:L] = 0
# rho[L:,L:] = 0
# rho[L:,:L] = A
# rho[:L,L:] = A.conj().T
# rev, revecs = np.linalg.eigh(rho)
# print(rev+0.5)
# 	
# print(stop)

# Op1 = Ops.c[3] @ Op
# Op2 = Op @ Ops.c[0] 
# print('A: ', np.trace(Op1.conj().T @ Op2) / np.trace(Op.conj().T @ Op))
# vec = Vectorized_Operator(Op, L).vec
# print('B: ', vec.conj().T @ Ops.S_didj[0,L+3] @ vec)
# print('C: ', np.trace(Ops.c[0].conj().T @ Ops.c[1]) / np.trace(Op.conj().T @ Op))
# print(stop)

# #Szj = (1/2 * np.identity(2**L) - Ops.n[6])
# #Op = U @ Szi @ U.conj().T 
# #Op = Ops.n[2]
# #Op = Op @ Op
# #Op = U @ Sz_j_t @ U.conj().T

# #Psi1 = Hevecs[:,0].reshape(-1,1)
# #Psi2 = Hevecs[:,1].reshape(-1,1)
# #Psi3 = Hevecs[:,2].reshape(-1,1)
# #Psi4 = Hevecs[:,3].reshape(-1,1)
# #Op = np.outer(Psi1, Psi1) + np.outer(Psi2, Psi2) + np.outer(Psi3, Psi3) + np.outer(Psi4, Psi4)

# Rev, Revecs = Spectrum_R_Op(Op, Ops)
# print(Rev)
# #for i in range(2*L):
# #	print("Rotation: ", i, "\n", np.round(np.abs(Revecs[:,i]), 6))
# #plt.matshow(np.abs(Revecs))
# #plt.show()

# vec = Vectorized_Operator(Op, L)
# #vec.show_dec()
# #vec.EE()
# #vec2 = np.moveaxis(Op.reshape((2,)*2*L), np.arange(L, 2*L), np.arange(1,2*L,2)).ravel() ## put dual and direct spaces next to each others
# #Information_Lattice(vec2, plot=True, title="Original")

# #OpEE(Op) ## Matches with the vectorial EE
# #vecO_rotate = vec.one_body_rot(Revecs.conj(), Ops)
# #vecO_rotate.show_dec()
# #vecO_rotate.EE()
# #Information_Lattice(vecO_rotate.vec, plot=True, title="Rotated")
# plt.show()

# print(kjhsef)

# ## Operators evolving in time
# ## ---------------------------------------------------------
t_list = np.arange(dt, T, dt)
spectrum_U_t  = np.zeros((len(t_list), 2*L))

U_dt = U.copy()
## Scaling with L
for L in [4,6,8]:
    #print("t: ", t)
    Li = 0 * Ops.sz[0]
    Lin = 1j*kron(H,-H) + kron(Li,Li) - 0.5 * kron(Li.conj().T @ Li, sp.sparse.identity(2**L))
    Lam = sp.sparse.expm(1 * Lin)
    ev, evec = np.linalg.eig(Lam)
    
    
## Test non-G gates    
# t_list = np.arange(dt, T, dt)
# spectrum_U_t  = np.zeros((len(t_list), 2*L))
# spectrum_Sz_t = np.zeros((len(t_list), 2*L))
# S  = np.zeros(len(t_list))

# g1 = sp.linalg.expm(-1j * Ops.n[2] @ Ops.n[3])
# g2 = sp.linalg.expm(-1j * Ops.n[3] @ Ops.n[4])
# g3 = sp.linalg.expm(-1j * Ops.n[4] @ Ops.n[5])
# nonG = [g1, g2, g3]

# for cpt, t in enumerate(t_list):
#     #print("t: ", t)
        
#    Op = U
#    ev, evecs = Spectrum_R_Op(Op, Ops)
#    ev = ev+1e-17
#    S[cpt] = - np.sum( np.log(np.abs(ev)) * np.abs(ev))
#    if True:
#    # if cpt == 0:
#        i = np.random.randint(3)
#        U = U @ nonG[i]
#    print(t, i, S[cpt], "\n")#, ev)  
        
# 	spectrum_U_t[cpt] = ev

# 	Op = Sz_j_t
# 	ev, evecs = Spectrum_R_Op(Op, Ops)
# 	spectrum_Sz_t[cpt] = ev

    U = U_dt @ U
# 	Sz_j_t = U_dt.conj().T @ Sz_j_t @ U_dt

# ## Save
# if save:
# 	f["t"] = t_list 
# 	f["U_t"] = spectrum_U_t 
# 	f["Sz_t"] = spectrum_Sz_t

# 	Op = H
# 	ev, evecs = Spectrum_R_Op(Op, Ops)
# 	f["H"] = ev 

# 	Hev, Hevecs = np.linalg.eigh(H)
# 	gs = Hevecs[:,0]
# 	Op =  np.outer(gs, gs)
# 	ev, evecs = Spectrum_R_Op(Op, Ops)
# 	f["density_matrix"] = ev 

# 	Qev, Qevecs = Spectrum_Q_St(gs, Ops)
# 	f["state"] = Qev 

# 	f.close()
