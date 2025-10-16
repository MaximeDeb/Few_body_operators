import numpy as np
import scipy as sp
import time

import sparseoperations as so
import symmpo as smpo


#---------------------------------------------------------------
def Sym_TensorDot(A, B, indices):
	''' 
	Given two tensors A and B, perform the tensor product AB along the dimensions specified in indices. L should be a set of diagonal matrices for each charge, which is multiplied
	between A and B (useful for Vidal's representation of the tensor network).
	For the TensorDot: 
		- Reshapes the tensors and defines the different blocks depending on the directions of the arrows for each legs. 
		- Multiplies the corresponding matrices by blocks, thus defining the new blocks to be reshaped.
		- If (reshapeBack == True): reshapes the result as a new tensor with the same convention as symmetric tensors: left - sigmas - sigmas' - right (sparse format)
		- Else: returns the blocks of the matrix. Can be useful to perform block SVDs.
	'''
	alpha = A.alpha
	x_ind, y_ind = np.array(indices[0]), np.array(indices[1])
	
	lA, lB = np.arange(A.nLegs), np.arange(B.nLegs)

	LlegsA, RlegsA = np.setdiff1d(lA, x_ind), x_ind  ## ( Legs on the left side , Legs to be summed )
	LlegsB, RlegsB = y_ind, np.setdiff1d(lB, y_ind)  ## ( Legs to be summed , Legs on the right side )

	A_blocks_mat, A_blocks_shapes = so.reshape_data_Tensors(A, LlegsA, RlegsA)
	B_blocks_mat, B_blocks_shapes = so.reshape_data_Tensors(B, LlegsB, RlegsB)

	## We determine the sector (symmetry rules) of each block given the considered legs.
	Rmask_arrowA, Rmask_sA = A.arrows[RlegsA] == 'i', A.legType[RlegsA] == 's'
	Lmask_arrowB, Lmask_sB = B.arrows[LlegsB] == 'i', B.legType[LlegsB] == 's'

	RblocksA = np.sum(A.coordinates[RlegsA] * (-1*Rmask_arrowA[:,None] + (~Rmask_arrowA)[:,None]) * ((A.alpha)*Rmask_sA[:,None] + (~Rmask_sA[:,None])),axis=0) 
	LblocksB = np.sum(B.coordinates[LlegsB] *  (Lmask_arrowB[:,None] - 1*(~Lmask_arrowB)[:,None]) * ((B.alpha)*Lmask_sB[:,None] + (~Lmask_sB[:,None])),axis=0) 

	SectorsA = np.unique(RblocksA) ## Determines all DIFFERENT blocks, and how many sub blocks are there in each block (in var. count) for A
	SectorsB = np.unique(LblocksB) ## Determines all DIFFERENT blocks, and how many sub blocks are there in each block (in var. count) for B
	Sectors = np.intersect1d(SectorsA, SectorsB) ## Taking the intersection is important, as all dimensions do not necessarily match.
	nSectors = len(Sectors)

	Ablock_sectors = RblocksA[None,:] == Sectors[:, None] ## Ablock_sectors[Sector, coord-of-A which is in Sector]
	Bblock_sectors = LblocksB[None,:] == Sectors[:, None] ## Bblock_sectors[Sector, coord-of-B which is in Sector]

	## We define, for every sector s, the corresponding matrices A and B made from a collection of subblocks
	L_BlockCoo, A_matShapesSect, nLsectA, nRsectA, IdBlocksA = so.construct_subBlocks_Sectors(A, LlegsA, RlegsA, A_blocks_shapes, Ablock_sectors, nSectors)
	R_BlockCoo, B_matShapesSect, nLsectB, nRsectB, IdBlocksB = so.construct_subBlocks_Sectors(B, LlegsB, RlegsB, B_blocks_shapes, Bblock_sectors, nSectors)

	## We check that the subsectors of the two matrices are facing each other correctly.
	L_BlockCoo, R_BlockCoo, A_matShapesSect, B_matShapesSect, emptySect = so.Check_subBlocks_Sectors(L_BlockCoo, R_BlockCoo, A_matShapesSect, B_matShapesSect, IdBlocksA, IdBlocksB)

	## reshape and defines the different blocks
	nBlocks_per_block = nLsectA * nRsectB
	Lphys_dims, Rphys_dims = int(np.sum(A.legType[LlegsA] != 'v')/2), int(np.sum(B.legType[RlegsB] != 'v')/2) 
	C = smpo.symmetric_Tensor(A.L, A.d, Lphys_dims + Rphys_dims, from_others=True, nLegs=len(LlegsA) + len(RlegsB), nSect=np.sum(nBlocks_per_block), alpha=A.alpha,
	data_as_tensors=A.data_as_tensors) ## Defines the new tensor
	C.arrows = np.array(tuple(A.arrows[LlegsA]) + tuple(B.arrows[RlegsB]))
	C.legType = np.array(tuple(A.legType[LlegsA]) + tuple(B.legType[RlegsB]))

	if A.data_as_tensors == False:
		if C.legType.shape[0] != 0:
			whvirt = C.legType == "v"
		else: ## If there is no leg in C!
			whvirt = ...

	nCoo = 0
	check = np.ones(np.sum(nBlocks_per_block), dtype=bool) ## Remove the empty blocks (might happen if Check_subBlocks leads to only unmatchings for some sectors)
	for s in range(nSectors):
		if (emptySect[s] != True):
			matA, Ablock_list_L, Ablock_list_R, Ablock_shape_L, Ablock_shape_R = so.construct_matrix_from_subBlocks(A_blocks_mat, A_matShapesSect[s], L_BlockCoo[s],
			Ablock_sectors[s])
			matB, Bblock_list_L, Bblock_list_R, Bblock_shape_L, Bblock_shape_R = so.construct_matrix_from_subBlocks(B_blocks_mat, B_matShapesSect[s], R_BlockCoo[s],
			Bblock_sectors[s])

			A_blocks = np.repeat(A.coordinates[LlegsA][:, Ablock_sectors[s]][:,Ablock_list_L], nRsectB[s],axis=1)
			B_blocks = np.tile(B.coordinates[RlegsB][:, Bblock_sectors[s]][:,Bblock_list_R], (1,nLsectA[s]))
			## Coordinates and shapes: Perform the correct reshape to tensor ! 
			C.coordinates[:,nCoo:(nCoo + nBlocks_per_block[s])] = np.concatenate((A_blocks, B_blocks), axis=0)

			A_blocks = np.repeat(A.shapes[LlegsA][:, Ablock_sectors[s]][:, Ablock_list_L], nRsectB[s], axis=1)
			B_blocks = np.tile(B.shapes[RlegsB][:, Bblock_sectors[s]][:, Bblock_list_R], (1, nLsectA[s]))
			C.shapes[:, nCoo:(nCoo + nBlocks_per_block[s])] = np.concatenate((A_blocks, B_blocks), axis=0)

			if A.data_as_tensors:
				rsh = C.shapes[:,nCoo:(nCoo + nBlocks_per_block[s])]
			else:
				rsh = C.shapes[whvirt, nCoo:(nCoo + nBlocks_per_block[s])]
	
			## Matrix C
			matC = matA @ matB
			## Divide the new matrix in a set of blocks, which are sorted according to the new coordinates C.coordinates
			cuts_rows, cuts_cols = np.cumsum(Ablock_shape_L), np.cumsum(Bblock_shape_R)
			split_matC = so.block_split(matC, cuts_rows[:-1], cuts_cols[:-1])

			C.data[nCoo:(nCoo + nBlocks_per_block[s])] = [split_matC[i].astype(complex).reshape(rsh[:,i]) for i in range(nBlocks_per_block[s])]
		else:
			check[nCoo:(nCoo + nBlocks_per_block[s])] = False ## Empty sector s.

		nCoo += nBlocks_per_block[s] ## new number of non-zero sectors defined for the sparse representation C

	## If some blocks are to be removed.
	C = smpo.MaskCoord(C, check)

	return C

#---------------------------------------------------------------
def Trace(A, B, conjA=False, conjB=False, verbose=False):
	'''
	Computes the trace of the MPO with the Pacman method, without requiring a matrix form.
	'''
	A, B = A.copy(), B.copy()

	PacMan = smpo.symmetric_Tensor(A.L, A.d, 0, from_others=True, nLegs=2, nSect=1, alpha=A.alpha, data_as_tensors=A.data_as_tensors)
	if A.alpha == -1: ## Change by starting from 0 and not from L
		PacMan.coordinates[:, 0] = (A.L,A.L)
		PacMan.LegsSectors = np.array((np.array((A.L,)),) + (np.array([A.L,]),), dtype=object)
	elif A.alpha == (A.L+1):
		PacMan.coordinates[:, 0] = (0,0)
		PacMan.LegsSectors = np.array((np.zeros((1,)),) + (np.array([0,]),), dtype=object)
	PacMan.shapes = np.ones((PacMan.nLegs,PacMan.nSect), dtype=int)
	PacMan.data[0] = np.ones((1,)*PacMan.nLegs)
	PacMan.arrows = np.array(('o', 'o'))
	PacMan.legType = np.array(('v', 'v'))

	for i in range(0,A.L):
		if conjA:
			for b in range(A.TN["B%s"%i].nSect):
				A.TN["B%s"%i].data[b] = A.TN["B%s"%i].data[b].conj()
			dimsA = [0,1,2]
			dimsB = [0,1,2]
		elif conjB:
			for b in range(B.TN["B%s"%i].nSect):
				B.TN["B%s"%i].data[b] = B.TN["B%s"%i].data[b].conj()
			dimsA = [0,1,2]
			dimsB = [0,1,2]
		else:
			A.TN["B%s"%i].legType[[1,2]] = A.TN["B%s"%i].legType[[2,1]] ## Don't transpose 
			dimsA = [0,1,2]
			dimsB = [0,2,1]

		PacA = Sym_TensorDot(PacMan, A.TN["B%s"%i], ([0], [0]))
		PacA.arrows[[1,2]] = 'o'
		PacMan = Sym_TensorDot(PacA, B.TN["B%s"%i], (dimsA, dimsB))

	if len(PacMan.data) == 0:
		return 0.0
	else:
		return np.sum(PacMan.data[0])

#---------------------------------------------------------------
def SitePacMan(O1, O2, conjA=False, conjB=False, left=True, right=True):
	'''
	Computes the trace of the MPO with the Pacman method, without requiring a matrix form.
	'''
	A, B = O1.copy(), O2.copy()
	L = A.L

	L_PacMan = smpo.symmetric_Tensor(A.L, A.d, 0, from_others=True, nLegs=2, nSect=1, alpha=A.alpha)
	R_PacMan = smpo.symmetric_Tensor(A.L, A.d, 0, from_others=True, nLegs=2, nSect=1, alpha=A.alpha)
	A_sect, B_sect = A.TN['B%s'%(L-1)].coordinates[-1][0], B.TN['B%s'%(L-1)].coordinates[-1][0]
	if A.alpha == -1:
		L_PacMan.coordinates[:, 0] = (L,L)
		L_PacMan.LegsSectors = np.array((np.array((L,)),) + (np.array([L,]),), dtype=object) ## SHOULD BE 0
		R_PacMan.coordinates[:, 0] = (A_sect,B_sect)
		R_PacMan.LegsSectors = np.array((np.array((A_sect,)),) + (np.array([B_sect,]),), dtype=object)
	elif A.alpha == (L+1):
		L_PacMan.coordinates[:, 0] = (0,0)
		L_PacMan.LegsSectors = np.array((np.array([0,]),) + (np.array([0,]),), dtype=object)
		R_PacMan.coordinates[:, 0] = (A_sect, B_sect)
		R_PacMan.LegsSectors = np.array((np.ones(1, dtype=int)*A_sect,) + (np.ones(1,dtype=int)*B_sect,), dtype=object)

	L_PacMan.shapes = np.ones((L_PacMan.nLegs, L_PacMan.nSect), dtype=int)
	L_PacMan.data[0] = np.ones((1,)*L_PacMan.nLegs, dtype=complex)
	L_PacMan.arrows = np.array(('o', 'o'))
	L_PacMan.legType = np.array(('v', 'v'))

	R_PacMan.shapes = np.ones((R_PacMan.nLegs, R_PacMan.nSect), dtype=int)
	R_PacMan.data[0] = np.ones((1,)*R_PacMan.nLegs, dtype=complex)
	R_PacMan.arrows = np.array(('i', 'i'))
	R_PacMan.legType = np.array(('v', 'v'))

	L_PM = {}
	R_PM = {}

	for i in range(L):
		L_A, L_B = A.TN["B%s"%i].copy(), B.TN["B%s"%i].copy()
		R_A, R_B = A.TN["B%s"%(L-1-i)].copy(), B.TN["B%s"%(L-1-i)].copy()
		if conjA:
			for b in range(L_A.nSect):
				L_A.data[b] = L_A.data[b].conj()
			for b in range(R_A.nSect):
				R_A.data[b] = R_A.data[b].conj()
			dimsL_A, dimsL_B = [0,1,2], [0,1,2] ## PacA @ B
			dimsR_A, dimsR_B = [1,2,3], [1,2,3] ## PacA @ B
		elif conjB:
			for b in range(L_B.nSect):
				L_B.data[b] = L_B.data[b].conj()
			for b in range(R_B.nSect):
				R_B.data[b] = R_B.data[b].conj()
			dimsL_A, dimsL_B = [0,1,2], [0,1,2] ## PacA @ B
			dimsR_A, dimsR_B = [1,2,3], [1,2,3] ## PacA @ B
		else:
			dimsL_A, dimsL_B = [0,1,2], [0,2,1] ## PacA @ B
			dimsR_A, dimsR_B = [1,2,3], [2,1,3] ## PacA @ B
			L_A.legType[[1,2]] = L_A.legType[[2,1]] ## Transpose
			R_A.legType[[1,2]] = R_A.legType[[2,1]] ## Transpose

		if left:
			L_PM[i] = L_PacMan ## Stores the left-PacMan at each site
			PacA = Sym_TensorDot(L_PacMan, L_A, ([0], [0]))
			PacA.arrows[[1,2]] = 'o'
			L_PacMan = Sym_TensorDot(PacA, L_B, (dimsL_A, dimsL_B))

		if right:
			R_PM[L-1-i] = R_PacMan ## Stores the right-PacMan at each site
			PacA = Sym_TensorDot(R_A, R_PacMan, ([3], [0]))
			PacA.arrows[[1,2]] = 'o'
			R_PacMan = Sym_TensorDot(R_B, PacA, (dimsR_B, dimsR_A))
			R_PacMan = smpo.swap_indices(R_PacMan, [1,0],[0,1])

		if (conjA==False) and (conjB==False): ## Transpose back
			L_A.legType[[1,2]] = L_A.legType[[2,1]] 
			R_A.legType[[1,2]] = R_A.legType[[2,1]]

	return L_PM, R_PM

#---------------------------------------------------------------
def OTOC(Szj, sites=None):
	''' 
	Given a local operator Sz_i(t), compte Tr(Sz_i(t) Sz_j Sz_i(t) Sz_j)
	Possible to select specific sites on which to compute the otoc.
	'''
	O = Szj.copy()
	otoc = np.zeros((O.L))
	if sites == None:
		sites = np.arange(O.L)

	L_PM, R_PM = SitePacMan(O, O) ## Computes the left and right PacMans and store them at every site
	norm = np.sum(Sym_TensorDot(L_PM[1], R_PM[0], ([0,1],[0,1])).data).real
	for i in sites: ### Product between tensor S_j^z(t) and S_i^z(0)
		O_i = smpo.Apply_Spin_Op("sz", O, i).TN['B%s'%i].copy() ## First define S_i^z(0)

		### < + >: multiply left-PacMan - Local Tensor - right-PacMan
		O_i.legType[[1,2]] = O_i.legType[[2,1]]
		A = Sym_TensorDot(L_PM[i], O_i, ([0], [0]))
		A.arrows[[1,2]] = 'o'
		O_i.legType[[1,2]] = O_i.legType[[2,1]]
		B = Sym_TensorDot(A, O_i, ([0,1,2], [0,2,1]))

		otoc[i] = np.sum(Sym_TensorDot(B, R_PM[i], ([0,1],[0,1])).data).real / norm

	return otoc

#---------------------------------------------------------------
def R_matrix(Op, unitary=False, opt=True):
	'''
	Computes the R matrix (see def) for a given MPO. 
	Naive scaling: L^3
	Optimal scaling: L^2
	'''
	O = Op.copy()
	R = np.zeros((2*O.L, 2*O.L), dtype=complex)
	norm_O = O.Norm()
	if opt: ## Optimal version (still minor optimizations possible but scaling is reached)
		_, R_PM_O = SitePacMan(O, O, left=False, conjA=True) ## This only stores identities in the correct sectors ..
		## Creates a version of O on which the fermionic sign is applied on the left OR on the right
		Osgn_L, Osgn_R = smpo.Apply_Spin_Op("sz", O, np.arange(O.L), side="L"), smpo.Apply_Spin_Op("sz", O, np.arange(O.L), side="R")
		for i in range(2*O.L):
			ind_i = i//2 
			side_i = "L" if i%2 == 0 else "R" ## Apply c on the left or on the right depending on i (direct or dual space)
			O_s = Osgn_R if side_i == "L" else Osgn_L
			### Apply <O|c+ or <O|c and c+|O> or c|O>
			if not unitary: ## Don't need this one if unitary
				O1_R = smpo.Apply_Fermionic_Op("c", O, ind_i, side_i, sign_side = "R") 
				L_PM_between, _ = SitePacMan(O1_R, O,   right=False, conjA=True) ## sign between c_i and c_j
			O1_L = smpo.Apply_Fermionic_Op("c", O, ind_i, side_i, sign_side = "L")
			L_PM_start, _   = SitePacMan(O1_L, O_s, right=False, conjA=True) 
			for j in range(i,2*O.L): ## Only the upper triangle
				if unitary and (i%2 == j%2): ## If unitary: O+O = identity  ->  <c+c> = 0.5
					continue 

				ind_j = j//2 
				side_j = "L" if j%2 == 0 else "R" ## Apply c on the left or on the right depending on j (direct or dual space)
				## Only one tensor on top and on bottom
				O1_loc = O1_L.TN['B%s'%ind_j].copy() ## c already applied
				for b in range(O1_loc.nSect):
					O1_loc.data[b] = O1_loc.data[b].conj()
				O2_loc = O.TN['B%s'%ind_j].copy()

				## Similar to the function apply_fermionic_op in symmpo.py, but we adapt it locally to fasten the computation
				## Apply c_j locally, without sign, it's already in c_i
				leg_nb  = 2 if side_j == "L" else 1
				inc_st  = 1 if side_j == "L" else 0  
				wh = O2_loc.coordinates[leg_nb,:] == inc_st 	
				O2_loc = smpo.MaskCoord(O2_loc, wh)    		

				## Careful: adapt sectors on the left
				## Adapt sectors on the right
				O2_loc.coordinates[leg_nb, :] = (1+inc_st)%2  ## 1 -> 0, 0 -> 1
				O2_loc.coordinates[-1,:] = O2_loc.coordinates[0,:] + O.alpha * O2_loc.coordinates[1,:] + O2_loc.coordinates[2,:] 
				O2_loc.LegsSectors = np.array(tuple([np.unique(O2_loc.coordinates[l,:]) for l in range(O2_loc.nLegs)]), dtype=object)

				##Adapt the sectors on the right pacman
				R_pac = R_PM_O[ind_j].copy()
				R_pac.coordinates[0,:] = R_pac.coordinates[0,:] - 1 if side_i == "L" else R_pac.coordinates[0,:] + O.alpha ## This one sees <O|c
				R_pac.coordinates[1,:] = R_pac.coordinates[1,:] - 1 if side_j == "L" else R_pac.coordinates[1,:] + O.alpha ## This one sees c|O>
				R_pac.LegsSectors = np.array(tuple([np.unique(R_pac.coordinates[l,:]) for l in range(R_pac.nLegs)]), dtype=object)

				### < + >: multiply left-PacMan - Local Tensor - right-PacMan
				PacA = Sym_TensorDot(L_PM_between[ind_j], O1_loc, ([0], [0])) if (side_i == side_j) else Sym_TensorDot(L_PM_start[ind_j], O1_loc, ([0], [0]))
				PacA.arrows[[1,2]] = 'o'
				B = Sym_TensorDot(PacA, O2_loc, ([0,1,2], [0,1,2]))

				R[O.L*(i%2) + ind_i, O.L*(j%2) + ind_j] = np.sum(Sym_TensorDot(B, R_pac, ([0,1],[0,1])).data) / norm_O
		if unitary:
			np.fill_diagonal(R[:Op.L,:Op.L], 0.5 * 2**Op.L / norm_O) ## Careful it depends on the sector of the operator, but we assume it is charge preserving (alpha = -1)
			np.fill_diagonal(R[Op.L:,Op.L:], 1 - 0.5 * 2**Op.L / norm_O) ## Careful it depends on the sector of the operator, but we assume it is charge preserving (alpha = -1)

	else: ## Naive version
		for i in range(2*O.L):
			ind_i = i//2 
			side_i = "L" if i%2 == 0 else "R" ## Apply c on the left or on the right depending on j (direct or dual space)
			### Apply <O|c+ or <O|c and c+|O> or c|O>
			O1 = O.copy()
			O1 = smpo.Apply_Fermionic_Op("c", O1, ind_i, side_i)
			for j in range(i,2*O.L):
				ind_j = j//2 
				side_j = "L" if j%2 == 0 else "R" ## Apply c on the left or on the right depending on j (direct or dual space)
				O2 = O.copy()
				O2 = smpo.Apply_Fermionic_Op("c", O2, ind_j, side_j) 

				## Compute (<O|c+)(c|O>) as Tr(O1+, O2)
				R[O.L*(i%2) + ind_i, O.L*(j%2)+ind_j] = Trace(O1, O2, conjA=True, conjB=False) / norm_O

	R = R + R.conj().T - np.diag(R.diagonal())

	return R
