import numpy as np
import scipy as sp

import linalg as la
import tebd as tebd

#---------------------------------------------------------------
class Symmetric_MPO:
	'''
	Contains the set of tensors representing an MPO in a Gamma-Lambda representation (it's actually a B-Lambda rep. for the Hasting's trick).
	L: system size (number of tensors)
	d: local dimension (dimension of the physical legs)
	q_alpha: symmetry sector in which the MPO lies (corresponding block of the operator)
	chi_max: maximum bond dimension of the largest symmetry sector (for each local tensor)
	phys_dims: number of physical dimensions for every local tensor
	bond_dims_sect: for each virtual dimension a_{i}, stores the dimension of every sub-vectorspace associated to every given symmetry sector
	'''
	def __init__(self, L, d, q_alpha, phys_dims, chi_max=None, chi_block=0, th_sing_vals=1e-8, data_as_tensors=True, alpha=-1, initial = None, is_symmetric=False,
	truncation_type="global"):
		'''
		Initialization of a symmetric MPO. Can only initialize an Identity operator, in a "sector" or "sparse" representation. Then, any local operator can easily be defined 
		starting from the identity MPO.
		'''
		self.L = L
		self.d = d
		self.q_alpha = q_alpha
		self.chi_max = chi_max
		self.chi_block = chi_block
		self.alpha = alpha
		self.phys_dims = phys_dims
		self.is_symmetric = False
		self.th_sing_vals = th_sing_vals 
		self.data_as_tensors = data_as_tensors
		self.truncation_type = truncation_type

		if initial == "Id":
			bond_dims_sect = init_virtual_sects(L, d, q_alpha, phys_dims, chi_max, initial, alpha)
			self.TN = {}
			for i in range(L):
				self.TN["B%s"%i] = symmetric_Tensor(L, d, phys_dims, bond_dims_sect[i], bond_dims_sect[i+1], initial, alpha=alpha,
				data_as_tensors=self.data_as_tensors, is_symmetric=self.is_symmetric)
				self.TN["Lam%s"%i] = symmetric_Lambda(L, d, self.TN["B%s"%i].LegsSectors[0], initial, chi_max = chi_max, chi_block=chi_block, glob_sect = q_alpha)

	#---------------------------------------------------------------
	def copy(self):
		'''
		Returns a copy of the MPO.
		'''
		B = Symmetric_MPO(self.L, self.d, self.q_alpha, self.phys_dims, chi_max=self.chi_max, chi_block=self.chi_block, th_sing_vals=self.th_sing_vals, alpha=self.alpha,
		initial="from_sect", is_symmetric = self.is_symmetric, data_as_tensors=self.data_as_tensors)
		B.TN = {}
		for i in range(self.L):
		     A = self.TN["B%s"%i]
		     B.TN["B%s"%i] = symmetric_Tensor(A.L, A.d, A.phys_dims, from_others=True, nLegs=A.nLegs, nSect=A.nSect, alpha=A.alpha, data_as_tensors=A.data_as_tensors) ## Defines the new tensor
		     B.TN["B%s"%i].coordinates 	= A.coordinates.copy()
		     B.TN["B%s"%i].data 	= A.data.copy()
		     B.TN["B%s"%i].shapes 	= A.shapes.copy()
		     B.TN["B%s"%i].LegsSectors 	= A.LegsSectors.copy()
		     B.TN["B%s"%i].arrows 	= A.arrows.copy()
		     B.TN["B%s"%i].legType 	= A.legType.copy()
		    
		     ### ? To be rechecked if it is initialized correctly! 
		     B.TN["Lam%s"%i] = symmetric_Lambda(A.L, A.d, A.LegsSectors[0], initial="from_sect", chi_max=self.chi_max, chi_block=self.chi_block, glob_sect=self.q_alpha) ## Defines the new tensor
		     A = self.TN["Lam%s"%i]
		     B.TN["Lam%s"%i].data 	= A.data.copy()
		     B.TN["Lam%s"%i].arrows 	= A.arrows.copy()
		     B.TN["Lam%s"%i].nSect 	= A.nSect
		     B.TN["Lam%s"%i].Lsect 	= A.Lsect
		     B.TN["Lam%s"%i].Rsect 	= A.Rsect
		
		return B

	#---------------------------------------------------------------
	def BondDim(self):
		'''
		Prints the bond dimensions of the tensors along the tensor network (sums the dimensions of every symmetry sector)
		'''
		## The bond dims. should be the same if we consider the full mat. or the sector! But the MPO scales with the norm
		maxind = -1
		left, right = np.zeros(self.L, dtype=int), np.zeros(self.L, dtype=int)
		for i in range(self.L):
			for sect in self.TN["B%s"%i].LegsSectors[0]:
				left[i] += self.TN["B%s"%i].shapes[0,:][self.TN["B%s"%i].coordinates[0,:][:] == sect][0]

			for sect in self.TN["B%s"%i].LegsSectors[-1]:
				right[i] += self.TN["B%s"%i].shapes[-1,:][self.TN["B%s"%i].coordinates[-1,:][:] == sect][0]
			newmax = np.array(np.where(self.TN["B%s"%i].shapes[0,:] * self.TN["B%s"%i].shapes[-1,:] == np.max(self.TN["B%s"%i].shapes[0,:] *self.TN["B%s"%i].shapes[-1,:])))[0,0]
			if newmax > maxind:
				maxind = newmax
				maxi = i
			
		print('--'.join(np.array([str(left[i]) + "+" + str(right[i]) for i in range(self.L)])))
		##print("Biggest block stored: ", self.TN["B%s"%maxi].shapes[[0,-1],maxind], " of coord	", self.TN["B%s"%maxi].coordinates[:,maxind], " in sect ",maxi)

	#---------------------------------------------------------------
	def entEntropy(self, l):
		'''
		Prints the entanglement entropy at the middle of the chain (TN should be a chain)
		Prints the entanglement entropy on site l if l==True     (TN should be a chain)
		'''
		print(l)
		if l==False:
			l = int(self.L/2) 
		else:
			l = int(l)
			s = np.concatenate(list( (self.TN["Lam%s"%l].data).values() ), axis=None ) #/ np.sqrt(np.math.factorial(self.L) / np.math.factorial(self.sector) / np.math.factorial(self.L - self.sector) )
			s = s / np.linalg.norm(s)
			S_entropy = - np.sum( s**2 * np.log(s**2) ) 

		return S_entropy

	#---------------------------------------------------------------
	def returnLambda(self, l):
		Lambda = []
		for key in self.TN['Lam%s'%l].data.keys():
			Lambda = np.concatenate((Lambda, self.TN['Lam%s'%l].data[key]))
		return np.sort(Lambda)

	#---------------------------------------------------------------
	def copyLambda(self):
		'''
		Returns an MPO with only the Lambda copied. 
		'''
		B = Symmetric_MPO(self.L, self.d, self.sector, self.phys_dims, initial="from_sect")
		B.TN = {}
		for i in range(self.L):
		     A = self.TN["B%s"%i]
		     B.TN["Lam%s"%i] = symmetric_Lambda(A.L, A.d, 0, initial="from_sect", site=None, chi_max=self.chi_max, chi_block=chi_block, glob_sect=self.sector) ## Defines the new tensor
		     A = self.TN["Lam%s"%i]
		     B.TN["Lam%s"%i].data       = A.data.copy()
		     B.TN["Lam%s"%i].arrows     = A.arrows.copy()
		     B.TN["Lam%s"%i].nSect      = A.nSect
		return B

	#---------------------------------------------------------------
	## We copy the MPO at integer time t
	def export(self, file):
		'''
		Export every attribute of an MPO in an external file.
		'''
		with h5py.File(file, 'w') as f:
			for attr in dir(self): 
				if not attr.startswith('__') and not callable(getattr(self, attr)) and (attr != "TN"):
					f.attrs[attr] = getattr(self, attr)

			for ii in range(self.L):
				f.create_dataset("B%s_coordinates"%ii, data=self.TN["B%s"%ii].coordinates)
				for jj in range(self.TN["B%s"%ii].data.shape[0]):
					f.create_dataset("B%s_data_%s"%(ii,jj), data=self.TN["B%s"%ii].data[jj])
				f.create_dataset("B%s_shapes"%ii, data=self.TN["B%s"%ii].shapes)
				for leg in range(self.TN["B%s"%ii].LegsSectors.shape[0]):
					f.create_dataset("B%s_LegsSectors_leg%s"%(ii,leg), data=str(self.TN["B%s"%ii].LegsSectors[leg]))
					f.create_dataset("B%s_arrows_leg%s"%(ii,leg), data=str(self.TN["B%s"%ii].arrows[leg]))
					f.create_dataset("B%s_legType_leg%s"%(ii,leg), data=str(self.TN["B%s"%ii].legType[leg]))

				for key in self.TN["Lam%s"%ii].data.keys():
					f.create_dataset("Lam%s_data_%s"%(ii,key), data=self.TN["Lam%s"%ii].data[key])
				f.create_dataset("Lam%s_nSect"%ii, data=self.TN["Lam%s"%ii].nSect)
				f.create_dataset("Lam%s_Lsect"%ii, data=self.TN["Lam%s"%ii].Lsect)
				f.create_dataset("Lam%s_Rsect"%ii, data=self.TN["Lam%s"%ii].Rsect)

		print("MPO exported to %s"%file)
	
	#---------------------------------------------------------------
	def toMatrix(self):
		''' 
		Transforms the MPO into a sparse matrix
		'''
		alpha = self.alpha
		if self.L>10:
			print("Matrix representation is too big")
			return(0)

		C = tebd.ApplyLambda(self.TN["Lam0"], self.TN["B0"])
		print("\n -- Merging tensors of the MPO -- ")
		for i in range(1,self.L):
			C = la.Sym_TensorDot(C, self.TN["B%s"%i], ([C.nLegs-1], [0]))
			## Trick for this specific case
			old_leg, new_leg, old_indices, new_indices = C.nLegs-3, int((C.nLegs-2)/2), np.arange(C.nLegs), np.arange(C.nLegs)
			old_indices[new_leg], old_indices[new_leg+1:old_leg+1] = new_indices[old_leg], new_indices[new_leg:old_leg]
			C = swap_indices(C, old_indices, new_indices)
			print("+ "*i + "o "*(self.L-i - 1))

		Lind = np.arange(0, 1+C.phys_dims)
		Rind = np.arange(1+C.phys_dims, 2+C.phys_dims*2)

		x = C.coordinates[1:1+C.phys_dims:,:][::-1,:]
		y = C.coordinates[1+C.phys_dims:-1:,:][::-1,:]

		x_coord = np.zeros(C.nSect, dtype=int) 
		y_coord = np.zeros(C.nSect, dtype=int) 
		data = np.zeros(C.nSect, dtype=complex)
		ind = np.arange(C.phys_dims, dtype=int)
		L, s = self.L, self.q_alpha
		if alpha == (L+1):
			Nst = 2**L
			states = np.zeros((Nst,L),dtype=int)
			for i in range(Nst):
			       states[i] = list(np.binary_repr(i,width=L))
			wh = np.where(np.sum(states,axis=1) == s)
			States_list = states[wh].astype(bool)[:,::-1]
			list_L = np.zeros(Nst, dtype=int)
			list_L[wh] = np.arange(0,States_list.shape[0],1)
			
			for i in range(C.nSect):
			       x_coord[i] = list_L[np.sum(np.power(2, ind * (x[:,i]!=0), dtype=np.int64) * x[:,i], axis=0)]
			       y_coord[i] = list_L[np.sum(np.power(2, ind * (y[:,i]!=0), dtype=np.int64) * y[:,i], axis=0)]
			       data[i] = np.sum(C.data[i])
			Cmat = sp.sparse.coo_matrix((data, (x_coord, y_coord)), shape=(len(wh[0]),len(wh[0])))
			
		elif (alpha == -1):
			for i in range(C.nSect):
				x_coord[i] = np.sum(np.power(2, ind * (x[:,i]!=0), dtype=np.int64) * x[:,i], axis=0)
				y_coord[i] = np.sum(np.power(2, ind * (y[:,i]!=0), dtype=np.int64) * y[:,i], axis=0)
				data[i] = np.sum(C.data[i])
			Cmat = sp.sparse.coo_matrix((data, (x_coord, y_coord)))
		else:
			print("toMatrix for alpha=%s is not implemented"%alpha)
	    
		return Cmat

	#---------------------------------------------------------------
	def Trace(self):
		PacMan = symmetric_Tensor(self.L, self.d, 0, from_others=True, nLegs=1, nSect=1, alpha=self.alpha)

		s0 = self.L if self.alpha == -1 else 0
		PacMan.coordinates[:, 0] = (s0)
		PacMan.LegsSectors = np.array((np.array((s0,)),), dtype=object)
		PacMan.shapes = np.ones((PacMan.nLegs, PacMan.nSect), dtype=int)
		PacMan.data[0] = np.ones((1,)*PacMan.nLegs)
		PacMan.arrows, PacMan.legType = np.array(['o']), np.array(['v'])
		
		## PacMan eats the MPO: o-- + + + + + + + ...
		for i in range(0,self.L):
			PacMan = la.Sym_TensorDot(PacMan, self.TN["B%s"%i], ([0], [0]))   # o-- +  -> o-- : Eats one site
			check  = np.zeros(PacMan.nSect, dtype=bool)
			## Check where to sum (sigma = sigma' for the trace)
			for n in range(PacMan.nSect):
				sigp = PacMan.coordinates[1,n]
				sig  = PacMan.coordinates[0,n]
				if sig == sigp:
					check[n] = 1
			PacMan.coordinates = PacMan.coordinates[-1,check].reshape(1,np.sum(check))
			PacMan.data = PacMan.data[check]

			coord, idx, inv = np.unique(PacMan.coordinates, return_index=True, return_inverse=True)
			PacMan.coordinates = coord.reshape(1,coord.shape[0])
			PacMan.LegsSectors = np.array((coord,), dtype=object)
			PacMan.shapes = PacMan.shapes[-1,check].reshape(1,np.sum(check))[:,idx]
			PacMan.arrows, PacMan.legType, PacMan.nLegs, PacMan.nSect = np.array(['o']), np.array(['v']), 1, PacMan.coordinates.shape[1]

			## Combine equal vector spaces
			dat = np.zeros(PacMan.nSect, dtype=object)
			for a in range(PacMan.nSect):
				equ_sect = np.where((inv==a))[1]
				for s in equ_sect: ## sum equiv sects.
					dat[a] += PacMan.data[s][0,0]
			PacMan.data = dat
		return PacMan.data[0][0]

	#---------------------------------------------------------------
	def Norm(self):
		return la.Trace(self, self, conjA=True)
        
#---------------------------------------------------------------
class symmetric_Gate:
	'''
	Trotter gate as a sparse tensor
	'''
	def __init__(self, phys_dims, dt, Params, gateType="Hamiltonian", model="Heis_nn", dag=False, alpha=-1, data_as_tensors=True, step=None, add_terms = None):
		self.L = Params["L"]
		self.d = Params["d"]
		self.phys_dims = phys_dims
		self.arrows = np.array(('i',) * self.phys_dims + ('o',) * self.phys_dims)
		self.nLegs = 2 * self.phys_dims
		self.legType = np.zeros(self.nLegs, dtype="str")
		self.legType[:] = 's'
		self.LegsSectors = np.zeros(2 * self.phys_dims, dtype=object)
		self.LegsSectors[:] = (np.array([0,1]),) * 2 * self.phys_dims
		self.alpha = alpha
		self.data_as_tensors = data_as_tensors

		if gateType == "Hamiltonian":
			Sp =  np.array(((0,0),(1.,0)))
			Sz =  np.array(((1.,0),(0,-1.))) * 0.5
			for key, value in Params.items():
				globals()[key] = value

			self.nSect = 6 ## Number of non-zero sectors

			if model == "Heis_nn":
				## Here, h_i = h_j and H_odd = H_even. "step" is thus not useful
				hi =  J/2 * ( np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp) ) + Jz * np.kron(Sz, Sz) 
				if periodT != 0: ## Floquet Hamiltonian, periodic Sz field h of period T
					hi += (hmean + hdrive * np.cos(2 * np.pi * t / periodT)) * (np.kron(np.identity(d), Sz) + np.kron(Sz, np.identity(d)))
			elif model == "IRLM":
				if step == "H0": ## H0 is defined as the gates acting on the impurity, different from the tight-binding in the bath
					hi = V * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + Uint * np.kron(Sz, Sz) 
				else:
					hi = gamma * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
			if add_terms is not None:
				hi += add_terms

			U = sp.linalg.expm(-1.j * hi * dt)

			self.coordinates = np.zeros((4,6), dtype=int)
			self.data = np.zeros(6, dtype=object)
			self.shapes = np.ones((4,6), dtype=int)

			self.coordinates[:,0] = [0,0,0,0]
			self.coordinates[:,1] = [0,1,0,1]
			self.coordinates[:,2] = [1,0,0,1]
			self.coordinates[:,3] = [0,1,1,0]
			self.coordinates[:,4] = [1,0,1,0]
			self.coordinates[:,5] = [1,1,1,1]

			legs = 4 if self.data_as_tensors else 2
			if dag == False:
				self.data[0] = np.ones((1,)*legs) * U[0,0]
				self.data[1] = np.ones((1,)*legs) * U[1,1]
				self.data[2] = np.ones((1,)*legs) * U[1,2]
				self.data[3] = np.ones((1,)*legs) * U[2,1]
				self.data[4] = np.ones((1,)*legs) * U[2,2]
				self.data[5] = np.ones((1,)*legs) * U[3,3]
			else:
				self.legType[:] = 'p'
				self.data[0] = np.ones((1,)*legs) * U[0,0].conj()
				self.data[1] = np.ones((1,)*legs) * U[1,1].conj()
				self.data[2] = np.ones((1,)*legs) * U[1,2].conj()
				self.data[3] = np.ones((1,)*legs) * U[2,1].conj()
				self.data[4] = np.ones((1,)*legs) * U[2,2].conj()
				self.data[5] = np.ones((1,)*legs) * U[3,3].conj() 

		elif gateType == "Swap":
			self.nSect = 4 ## Number of non-zero sectors

			self.coordinates = np.zeros((4,4), dtype=int)
			self.data = np.zeros(4, dtype=object)
			self.shapes = np.ones((4,4), dtype=int)

			self.coordinates[:,0] = [0,0,0,0]
			self.coordinates[:,1] = [1,0,0,1]
			self.coordinates[:,2] = [0,1,1,0]
			self.coordinates[:,3] = [1,1,1,1]

			legs = 4 if self.data_as_tensors else 2

			self.data[0] = np.ones((1,)*legs) * 1
			self.data[1] = np.ones((1,)*legs) * 1
			self.data[2] = np.ones((1,)*legs) * 1
			self.data[3] = np.ones((1,)*legs) * 1
		else:
			print("Unknown gate")

#---------------------------------------------------------------
class symmetric_Lambda:
	'''
	Lambda matrix in the Gamma-Lambda representation of a Tensor network. A Lambda is defined for every non-zero sub-vector space of the right leg of each tensor. 
	'''
	def __init__(self, L, d, sect, initial="Random", chi_max=None, chi_block=0, glob_sect=0): 
		self.chi_max = chi_max
		self.chi_block = chi_block
		if (initial == "Id") or (initial[0] == "S"):
			self.nSect = sect.shape[0]
			self.Lsect = sect
			self.Rsect = sect
			self.data = {}
			self.arrows = np.array(('i','o'))

			#norm = np.math.factorial(L) / (np.math.factorial(L-glob_sect) * np.math.factorial(glob_sect))
			for i in self.Lsect:
				self.data[i] = np.ones((1), dtype=float) 

		elif initial == "from_SVD":
			self.data = {}
			self.arrows = np.array(('i','o'))

	#---------------------------------------------------------------
	def def_sects(self, sect):
			self.nSect = len(sect)
			self.Lsect = sect
			self.Rsect = sect

#---------------------------------------------------------------
class symmetric_Tensor:
	'''
	Attributes of the tensor T:

	.nLegs: Number of legs (vector spaces) of T.
	.nSect: Number of non-zero blocks. (sparse representation)
	.arrows: Array: (nLegs)
	.coordinates: Tensor storing the non-zero elements by blocks
	.d:
	.data:
	.L:
	.LegsSectors:
	.phys_dims:
	.sect_L: Labels (mask (l,lp)) of the non_zero sectors of the LEFT  virtual leg.
	.sect_R: Labels (mask (l,lp)) of the non_zero sectors of the RIGHT virtual leg.
	.shapes:
	.legType:  Tells if the corresponding leg refers to a virtual ('v'), sigma ('s') or sigma' ('p') dimension.
	.data_as_tensors: True: data is stored as an nLegs-legged tensor. False: matrix (only 1d)
	''' 
	def __init__(self, L, d, phys_dims, a_L=None, a_R=None, initial=None, verbose=False, from_others=None, nLegs=None, nSect=None, alpha=-1, is_symmetric=False,
	data_as_tensors=True):
		self.L = L
		self.d = d
		self.phys_dims = phys_dims
		self.data_as_tensors = data_as_tensors
		if from_others:
			self.nLegs = nLegs
			self.nSect = nSect
			self.legType = np.zeros(nLegs, dtype="str")

			self.coordinates = np.zeros((self.nLegs, self.nSect),  dtype=int) ## sectL, sig..., sigp..., sectR, for the first dim. the second dim is the number of non
			self.data = np.zeros(self.nSect, dtype=object) ## block matrix for each non zero sector
			self.shapes = np.ones((self.nLegs, self.nSect), dtype=int) ## shapes of each block matrix (nLegs for first index: number of states in the vect. space. of each leg)
			self.alpha = alpha
			self.is_symmetric = is_symmetric

		if initial != None:
			self.alpha = alpha
			self.is_symmetric = is_symmetric
			self.nLegs = 2 + 2*phys_dims
			self.legType = np.zeros(self.nLegs, dtype="str")
			self.legType[np.array((0,-1))] = 'v'
			## Non-zero sectors
			Lsect = np.where(a_L) ## l: 1st dim. of sect_L, l': 2nd dim. of sect_L
			Rsect = np.where(a_R)
			## Indices of the legs for the physical legs (sigma and sigma')
			self.legType[np.arange(1,phys_dims+1)] = 's'
			self.legType[np.arange(phys_dims+1, 2*phys_dims+1)] = 'p'

			self.arrows = np.array(('i',) + ('i',)*2*phys_dims + ('o',))

			## Defines the sectors in a_R connected to the ones in a_L via the sigmas
			loc_sig = np.arange(0,d)
			if (alpha == L+1):
				conn_l  = Lsect[0][(...,) + (None,)*phys_dims] ## l_R  =  l_L + \sum sigma
				conn_lp = Lsect[1][(...,) + (None,)*phys_dims] ## lp_R = lp_L + \sum sigma'
				for i in range(phys_dims): ## We sum each sigma to l, and each sigma' to l'
					sig_i = loc_sig[(None,) + (None,)*i + (...,) + (None,)*(phys_dims-i-1)]
					conn_l  = conn_l  + sig_i
					conn_lp = conn_lp + sig_i
				## We check now if the connected sectors are allowed by the symmetry constraints
				mask  = np.in1d(conn_l,  Rsect[0]).reshape(conn_l.shape)  ## check all connected l allowed in sectors l_R:      SHAPE: l, sig ...
				maskp = np.in1d(conn_lp, Rsect[1]).reshape(conn_lp.shape) ## check all connected l' allowed in sectors l'_R:    SHAPE: l', sig' ...
				## Check what are the possible options matching for connect_l AND connect_lp: shape: (l,l'), sig..., sig'... REMARK: (l,l') is given as one number i, which is a common entry for sect_L[0][i] and sect_L[1][i]
				allowed_combinations = mask[(slice(None),) + (...,) + (None,)*phys_dims] * maskp[(slice(None),) + (None,)*phys_dims + (...,)] 
				if (initial == "Id") or (initial[0] == "S"):
					allowed_combinations = allowed_combinations * (conn_l[(slice(None),) + (...,) + (None,)*phys_dims] == conn_lp[(slice(None),) + (None,)*phys_dims + (...,)])
			elif (alpha == -1): 
				conn_l_min_lp = Lsect[0][(...,) + (None,)*(2*phys_dims)] ## l_R  =  l_L + \sum sigma
				for i in range(phys_dims): ## We sum each sigma to l, and each sigma' to l'
					sig_i = loc_sig[(None,) + (None,)*i + (...,) + (None,)*(2*phys_dims-i-1)]
					sig_ip = loc_sig[(None,) + (None,)*(phys_dims+i) + (...,) + (None,)*(phys_dims-i-1)]
					conn_l_min_lp = conn_l_min_lp + sig_i - sig_ip
				allowed_combinations = np.in1d(conn_l_min_lp,  Rsect[0]).reshape(conn_l_min_lp.shape)  ## check all connected l allowed in sectors l_R:      SHAPE: l, sig ...

			mask_conn = np.array(np.where(allowed_combinations)) ## mask_conn: [0]: always vect. space. (l,l')_R. [1]->[phys_dims]:sig, [phys_dims+1]->[2*phys_dims+1]:sig'
			#mask_conn = mask_conn[:,:1] ## to have a bond dimension 1 if it's identity!

			mask_Lsect  = (mask_conn[0,:],)  ## Labels of the allowed sectors (l,l')_R on the right
			mask_sig, mask_sigp = (), ()
			for i in range(phys_dims):
				mask_sig  += (mask_conn[1+i],)  	    ## We add all the allowed possibilities of all sigmas
				mask_sigp += (mask_conn[1+phys_dims+i],)    ## We add all the allowed possibilities of all sigmas'

			## Assign to each pair of connected sectors the matrix of the correct size (depending on degeneracies of the symmetry sectors) (defines the coordinates)
			self.nSect = mask_Lsect[0].shape[0] ## Number of non-zero sectors
			self.coordinates = np.zeros((phys_dims*2+2, self.nSect),  dtype=int) ## sectL, sig..., sigp..., sectR
			self.data = np.zeros(self.nSect, dtype=object)
			self.shapes = np.ones((self.nLegs,self.nSect), dtype=int)
			dims = np.array( ((L+1)**2,)+(2,)*(2*phys_dims)+((L+1)**2,))  ## sectL, sig..., sigp..., sectR
			for i in range(self.nSect):
				loc_sig, loc_sigp = (), ()
				for j in range(phys_dims):
					loc_sig  = loc_sig  +  (mask_sig[j][i],)
					loc_sigp = loc_sigp + (mask_sigp[j][i],)

				if alpha == L+1:
					lL, lpL = Lsect[0][mask_Lsect[0][i]],  Lsect[1][mask_Lsect[0][i]] 
					lR, lpR = conn_l[(mask_Lsect[0][i],) + loc_sig], conn_lp[(mask_Lsect[0][i],) + loc_sigp] 
					self.coordinates[0, i] = lL * (L+1) + lpL ## sectL
					self.coordinates[-1, i] = lR * (L+1) + lpR ## sectR
					degenL, degenR = a_L[lL, lpL], a_R[lR, lpR]
				elif alpha == -1:
					l_min_lp_L = Lsect[0][mask_Lsect[0][i]]
					l_min_lp_R = conn_l_min_lp[(mask_Lsect[0][i],) + loc_sig + loc_sigp]
					self.coordinates[0, i] = l_min_lp_L ## sectL
					self.coordinates[-1, i] = l_min_lp_R ## sectR
					degenL, degenR = a_L[l_min_lp_L], a_R[l_min_lp_R]

				self.coordinates[1:phys_dims+1, i]  = np.array(loc_sig) ## sig
				self.coordinates[phys_dims+1:-1, i] = np.array(loc_sigp) ## sigp
				if (initial == "Id") or (initial[0] == "S"):
					self.data[i] = np.ones((1,)*self.nLegs) if self.data_as_tensors else np.ones((1,1))
			## Blocks
			self.LegsSectors = np.array((np.unique(self.coordinates[0,:]),) + (np.array([0,1]),) * 2*phys_dims + (np.unique(self.coordinates[-1,:]),), dtype=object)

	#---------------------------------------------------------------
	def copy(self):
		B = symmetric_Tensor(self.L, self.d, self.phys_dims, from_others=True, nLegs=self.nLegs, nSect=self.nSect, alpha=self.alpha, data_as_tensors=self.data_as_tensors) ## Defines the new tensor
		B.coordinates 	= self.coordinates.copy()
		B.data 		= self.data.copy()
		B.shapes 	= self.shapes.copy()
		B.LegsSectors 	= self.LegsSectors.copy()
		B.arrows 	= self.arrows.copy()
		B.legType 	= self.legType.copy()

		return B

#---------------------------------------------------------------
def importMPO(file):
	'''
	Import an MPO from an hdf5 file.
	'''
	with h5py.File(file, "r") as f:
		globals().update(f.attrs)
		MPO  = Symmetric_MPO(L, d, q_alpha, phys_dims, chi_max, chi_block=0, alpha = alpha, initial = None, data_as_tensors = data_as_tensors, th_sing_vals = th_sing_vals)
		MPO.TN = {}
		for i in range(L):
			## Initialize classes
			MPO.TN["B%s"%i]   = symmetric_Tensor(L, d, phys_dims, alpha=alpha, data_as_tensors = data_as_tensors)
			MPO.TN["Lam%s"%i] = symmetric_Lambda(L, d, 0, initial="yolo", chi_max=chi_max, chi_block=chi_block)

			### import Bs
			MPO.TN["B%s"%i].coordinates = np.array((f["B%s_coordinates"%i]))
			MPO.TN["B%s"%i].nLegs =  MPO.TN["B%s"%i].coordinates.shape[0]
			MPO.TN["B%s"%i].nSect =  MPO.TN["B%s"%i].coordinates.shape[1]
			MPO.TN["B%s"%i].data = np.zeros((MPO.TN["B%s"%i].nSect), dtype=object)
			for j in range(MPO.TN["B%s"%i].nSect):
				MPO.TN["B%s"%i].data[j] = np.array(f["B%s_data_%s"%(i,j)])
			MPO.TN["B%s"%i].shapes = np.array(f["B%s_shapes"%i])
			MPO.TN["B%s"%i].alpha = alpha 
			MPO.TN["B%s"%i].LegsSectors = np.array((np.unique(MPO.TN["B%s"%i].coordinates[0,:]),) + (np.array([0,1]),) * 2*phys_dims + (np.unique(MPO.TN["B%s"%i].coordinates[-1,:]),), dtype=object)

			MPO.TN["B%s"%i].arrows      = np.zeros((MPO.TN["B%s"%i].nLegs), dtype=object)
			MPO.TN["B%s"%i].legType     = np.zeros((MPO.TN["B%s"%i].nLegs), dtype=object)
			for leg in range(MPO.TN["B%s"%i].nLegs):
				MPO.TN["B%s"%i].arrows[leg]  = str(np.array(f["B%s_arrows_leg%s"%(i,leg)]))[2:-1]
				MPO.TN["B%s"%i].legType[leg] = str(np.array(f["B%s_legType_leg%s"%(i,leg)]))[2:-1]  #np.array(f["B%s_legType_leg%s"%(ii,leg)])

			## import Lambdas
			MPO.TN["Lam%s"%i].arrows = np.array(('i','o'))
			MPO.TN["Lam%s"%i].data = {}
			for sect in np.array(f["Lam%s_Lsect"%i]):
				MPO.TN["Lam%s"%i].data[sect] = np.array(f["Lam%s_data_%s"%(i,sect)]).astype(complex)
			MPO.TN["Lam%s"%i].nSect = np.array((f["Lam%s_nSect"%i]))
			MPO.TN["Lam%s"%i].Lsect = np.array((f["Lam%s_Lsect"%i]))
			MPO.TN["Lam%s"%i].Rsect = np.array((f["Lam%s_Rsect"%i]))

	print("MPO imported")
	return MPO

#---------------------------------------------------------------
def init_virtual_sects(L, d, s, phys_dims, chi_max, initial="Random", alpha=-1):
	''' 
	Given an MPO in a fixed symmetry sector s, return the dimensions of the "global" symmetry sector for each local virtual leg. The available sectors for an MPO on L sites are
	written in a matrix form (l,l')
	'''
	loc_sig = np.arange(d) ## Local physical space
	## Stores the sub-vect. spaces for each virtual leg along the chain, from a_0 to a_L
	## We initialize all bond dimensions to 0
	if alpha == (L+1):
		list_LR = np.zeros((L+1, L+1, L+1), dtype=int)
		list_RL = np.zeros((L+1, L+1, L+1), dtype=int)
		list_intersec = np.zeros((L+1, L+1, L+1), dtype=int)
		## Bondary conditions: on left and right, we impose the sector
		list_LR[0, 0, 0] = 1  ## Initial sector (always (0,0))
		list_RL[L, s, s] = 1  ## Final sector
	elif alpha == -1:
		list_LR = np.zeros((L+1, 2*L+1), dtype=int)
		list_RL = np.zeros((L+1, 2*L+1), dtype=int)
		list_intersec = np.zeros((L+1, 2*L+1), dtype=int)

		list_LR[0, L] = 1  ## Initial sector: always l-l'=0
		list_RL[L, L] = 1  ## Final sector: always 0

	## We go through every site of the MPO, from L->R
	for i in range(L):
		## Stores all the pair of indices (l,l') for which a_i has a non-zero degeneracy. We ravel, such that l,l' -> l"
		llp = np.where(list_LR[i].ravel())[0]
		d_llp = list_LR[i].ravel()[llp] ## Stores the corresponding degeneracy

		## Applies sigma to l and sigma' to l' to define the new available states (lists grows by a factor d**2)
		avail_llp = llp[(...,) + (None,)*(2*phys_dims)]
		degen = d_llp[(...,) + (None,)*(2*phys_dims)] ## Give the same shape to old degeneracies
		for j in range(phys_dims): ## We sum each sigma to l, and each sigma' to l'
			sig_i = loc_sig[(None,) + (None,)*j + (...,) + (None,)*(2*phys_dims - j - 1)]
			sig_ip = loc_sig[(None,) + (None,)*(phys_dims+j) + (...,) + (None,)*(phys_dims - j - 1)]
			avail_llp = avail_llp + sig_i + alpha * sig_ip
			degen = degen + 0*sig_i + 0*sig_ip
		avail_llp = avail_llp.reshape(llp.shape[0], -1).ravel()
		degen = degen.reshape(llp.shape[0], -1).ravel() ## Give the same shape to old degeneracies

		if alpha == (L+1):
			mask = (avail_llp%(L+1) <= (L)) * (avail_llp//(L+1) <= (L))  ## Cannot go further to the right / lower than the starting point (we can not add more particles than the sector imposes)
			if (initial == "Id") or (initial[0] == "S"): ## NOT OPTIMIZED
				mask = mask * (avail_llp//(L+1) == avail_llp%(L+1))
		elif alpha == -1:
			mask = (avail_llp >= (0))  ## Cannot go further to the right / lower than the starting point (we can not add more particles than the sector imposes)
			if (initial == "Id") or (initial[0] == "S"): ## NOT OPTIMIZED
				mask = avail_llp == L
		degen = degen[mask]
		avail_llp = avail_llp[mask]

		## Check how many different sectors are now spanned
		u, ind = np.unique(avail_llp, return_inverse=True)
		new_degen = np.zeros(u.shape, dtype=int)
		for j in range(u.shape[0]): ## For each different new defined sector, we sum the number of previous degeneracies of the sect. generating this new one.
			new_degen[j] = np.minimum(np.max(np.array((0,np.sum(degen[u[ind]==u[j]])))), chi_max)

		if alpha == (L+1):
			new_l, new_lp = u // (L+1), u % (L+1)
			list_LR[i+1][new_l, new_lp] += new_degen 
		elif alpha == -1:
			list_LR[i+1][u] += new_degen 
		

	## Now, we go from R->L to impose the correct symmetry sector
	for i in range(L,0,-1):
		llp = np.where(list_RL[i].ravel())[0]
		d_llp = list_RL[i].ravel()[llp] 

		avail_llp = llp[(...,) + (None,)*(2*phys_dims)]
		degen = d_llp[(...,) + (None,)*(2*phys_dims)] ## Give the same shape to old degeneracies
		for j in range(phys_dims): ## We sum each sigma to l, and each sigma' to l'
			sig_i = loc_sig[(None,) + (None,)*j + (...,) + (None,)*(2*phys_dims - j - 1)]
			sig_ip = loc_sig[(None,) + (None,)*(phys_dims+j) + (...,) + (None,)*(phys_dims - j - 1)]
			avail_llp  = avail_llp  - sig_i - alpha * sig_ip
			degen = degen + 0*sig_i + 0*sig_ip
		avail_llp = avail_llp.reshape(llp.shape[0], -1)
		degen = degen.reshape(llp.shape[0], -1)#.ravel() ## Give the same shape to old degeneracies

		if alpha == (L+1):
			mask = (avail_llp%(L+1) <= s) * (avail_llp >= 0)  ## Cannot go further to the right / lower than the starting point (we can not add more particles than the desired sector imposes)
			if (initial == "Id") or (initial[0] == "S"): ## NOT OPTIMIZED
				mask = mask * (avail_llp//(L+1) == avail_llp%(L+1))
		elif alpha == -1:
			if (initial == "Id") or (initial[0] == "S"): ## NOT OPTIMIZED
				mask = avail_llp == L
		degen = degen[mask]  
		avail_llp = avail_llp[mask]

		## Check how many different sectors are now spanned
		u, ind = np.unique(avail_llp, return_inverse=True)
		new_degen = np.zeros(u.shape, dtype=int)
		for j in range(u.shape[0]): ## For each different new defined sector, we sum the number of previous degeneracies of the sect. generating this new one.
			new_degen[j] = np.minimum(np.max(np.array((0,np.sum(degen[u[ind]==u[j]])))), chi_max)

		if alpha == (L+1):
			new_l, new_lp = u // (L+1), u % (L+1)
			list_RL[i-1][new_l, new_lp] += new_degen 
		elif alpha == -1:
			list_RL[i-1][u] += new_degen 

	for l in range(len(list_RL)):
		list_intersec[l] = np.minimum(list_LR[l], list_RL[l])
		if alpha == (L+1):
			if (initial == "Id") or (initial[0] == "S"):
				list_intersec[l] = np.minimum(list_intersec[l], 1)

	return list_intersec

#---------------------------------------------------------------
def swap_indices(A, old, new): ## Swap indices of a tensor
	A.coordinates[new,:] = A.coordinates[old,:]
	A.shapes[new,:] = A.shapes[old,:]
	A.arrows[new] = A.arrows[old] 
	A.legType[new] = A.legType[old] 
	A.LegsSectors[new] = A.LegsSectors[old] 
	if A.data_as_tensors:
		for b in range(A.nSect):
			A.data[b] = np.moveaxis(A.data[b], old, new)

	return A

#---------------------------------------------------------------
def MaskCoord(A, mask):
	A.coordinates = A.coordinates[:,mask]
	A.data	      = A.data[mask]
	A.shapes      = A.shapes[:,mask]
	A.nSect       = np.sum(mask)
	A.LegsSectors = np.array(tuple([np.unique(A.coordinates[i,:]) for i in range(A.nLegs)]), dtype=object)
	return A

#---------------------------------------------------------------
def Apply_Fermionic_Op(A, Op, i, side, sign_side="L"):
	'''
	Applies a fermionic operator A (c or c+) on the tensor B_i of a symmetric MPO O.
	'''
	O = Op.copy()

	leg_nb = 2 if side == "L" else 1 ## Acting from the left is acting on the dual space, if the op acts on a state by its direct space
	inc_st  = 1 if ((A == "c" and side == "L") or (A == "c+" and side == "R")) else 0  ## c+O and Oc are similar

	wh = O.TN['B%s'%i].coordinates[leg_nb,:] == inc_st ## Check elements which survive: sigma=1 for c and sigma=0 for c+
	O.TN['B%s'%i] = MaskCoord(O.TN['B%s'%i], wh)    ## Kill the elements which does not survive

	O.TN['B%s'%i].coordinates[leg_nb, :] = (1+inc_st)%2  ## 1 -> 0, 0 -> 1
	O.TN['B%s'%i].coordinates[-1,:] = O.TN['B%s'%i].coordinates[0,:] + O.alpha * O.TN['B%s'%i].coordinates[1,:] + O.TN['B%s'%i].coordinates[2,:] ## We remove (or add) the charges
	O.TN['B%s'%i].LegsSectors = np.array(tuple([np.unique(O.TN['B%s'%i].coordinates[l,:]) for l in range(O.TN['B%s'%i].nLegs)]), dtype=object)
	for j in range(i+1, O.L): ## We spread the word that a particle has been removed or added
		if A == "c":
			O.TN['B%s'%j].coordinates[0,:] = O.TN['B%s'%j].coordinates[0,:] - 1 if side == "L" else O.TN['B%s'%j].coordinates[0,:] + O.alpha
		elif A == "c+": 
			O.TN['B%s'%j].coordinates[0,:] = O.TN['B%s'%j].coordinates[0,:] + 1 if side == "L" else O.TN['B%s'%j].coordinates[0,:] - O.alpha
		O.TN['B%s'%j].coordinates[-1,:] = O.TN['B%s'%j].coordinates[0,:] + O.alpha * O.TN['B%s'%j].coordinates[1,:] + O.TN['B%s'%j].coordinates[2,:]
		O.TN['B%s'%j].LegsSectors = np.array(tuple([np.unique(O.TN['B%s'%j].coordinates[l,:]) for l in range(O.TN['B%s'%j].nLegs)]), dtype=object)

	## Fermionic sign
	site_sign = np.arange(i) if sign_side == "L" else np.arange(i+1, O.L) ## Depends if we want c_i or c_i c_j wiht j>i
	O = Apply_Spin_Op("sz", O, site_sign, side=side) ## Apply Sz on the site i

	return O

#---------------------------------------------------------------
def Apply_Spin_Op(A, Op, sites, side="L"):
	''' 
	Applies a spin operator A (Sz) on the tensor B_i of a symmetric MPO O.
	'''
	O = Op.copy()
	leg_nb = 2 if side == "L" else 1 ## Acting from the left is acting on the dual space, if the op acts on a state by its direct space
	if A == "sz":
		if type(sites) == int:
			sites = [sites,]
		for j in sites:
			for n in np.where(O.TN['B%s'%j].coordinates[leg_nb,:] == 1)[0]:
				O.TN['B%s'%j].data[n] = O.TN['B%s'%j].data[n] * (-1)
	return O

#---------------------------------------------------------------
