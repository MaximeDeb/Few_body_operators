import numpy as np
import scipy as sp

import sparseoperations as so
import symmpo as smpo
import linalg as la

#---------------------------------------------------------------
def ApplyGate(Szj, U, Udag, l1, l2, bothSides=True):
	PhiB = la.Sym_TensorDot(Szj.TN["B%s"%l1], Szj.TN["B%s"%l2], ([3],[0]))
	old_indices, new_indices = [0,1,3,2,4,5], [0,1,2,3,4,5] ## Swap back indices
	PhiB = smpo.swap_indices(PhiB, old_indices, new_indices)

	## UPhi = U * PhiB
	UPhiB = la.Sym_TensorDot(U, PhiB, ([2,3],[1,2]))
	old_indices, new_indices = [2,0,1,3,4,5], [0,1,2,3,4,5] ## Swap back indices
	UPhiB = smpo.swap_indices(UPhiB, old_indices, new_indices)

	if bothSides:
		## UPhiBU = U * PhiB * U+
		UPhiBU = la.Sym_TensorDot(UPhiB, Udag, ([3,4], [2,3]))
		old_indices, new_indices = [0,1,2,4,5,3], [0,1,2,3,4,5] ## Swap back indices
		UPhiBU = smpo.swap_indices(UPhiBU, old_indices, new_indices)
	else:
		UPhiBU = UPhiB

	## Splits the tensors
	Szj, _, _, _, s_disc = Sym_Canonic(UPhiBU, Szj, l1, l2, truncation_type=Szj.truncation_type)

	return Szj, s_disc

#---------------------------------------------------------------
def CheckTruncation(M, l1, l2):
	l = l1
	diff = np.setdiff1d(np.array(list(M.TN["Lam%s"%l].data.keys())), M.TN["B%s"%l].LegsSectors[0])
	## /!\ Achtung: it propagates. if we delete the legs of B_l-1 on the right, we might suppress some legs on the left, which will change the left lambda too, etc. 
	while (diff.shape[0] != 0): ## Meaning that B1 lost sectors on the left, we have to remove these sectors in Lambda and in the B on the left of Lambda 
		M.TN["Lam%s"%l].def_sects(M.TN["B%s"%l].LegsSectors[0])
		mask = np.ones(M.TN["B%s"%(l-1)].nSect, dtype=bool)
		for difkey in diff:
			del M.TN["Lam%s"%l].data[difkey] 
			mask *= (M.TN["B%s"%(l-1)].coordinates[-1]) != difkey
		M.TN["B%s"%(l-1)] = Check_Coord(M.TN["B%s"%(l-1)], mask)
		l -= 1
		diff = np.setdiff1d(np.array(list(M.TN["Lam%s"%l].data.keys())), M.TN["B%s"%l].LegsSectors[0])
	## We do it on the right also, and propagate to the right
	l = l2
	diff = np.setdiff1d(M.TN["B%s"%l].LegsSectors[-1], np.array(list(M.TN["Lam%s"%(l+1)].data.keys())))
	while (diff.shape[0] != 0): ## Meaning that B1 lost sectors on the left, we have to remove these sectors in Lambda and in the B on the left of Lambda 
		M.TN["Lam%s"%(l+1)].def_sects(M.TN["B%s"%l].LegsSectors[-1])
		mask = np.ones(M.TN["B%s"%(l+1)].nSect, dtype=bool)
		for difkey in diff:
			del M.TN["Lam%s"%(l+1)].data[difkey] 
			mask *= (M.TN["B%s"%(l+1)].coordinates[0]) != difkey
		M.TN["B%s"%(l+1)] = Check_Coord(M.TN["B%s"%(l+1)], mask)
		l += 1
		diff = np.setdiff1d(M.TN["B%s"%l].LegsSectors[-1], np.array(list(M.TN["Lam%s"%(l+1)].data.keys())))
	return M

#---------------------------------------------------------------
def truncation(S, Vh, chi, chi_block, truncation_type, totStates, largestSect=None):
	'''
	truncation.
	'''
	trunc = False ## Check if we truncated
	if truncation_type == "block_threshold":
		if S[largestSect].shape[0] > chi:
			chi = np.minimum(len(S[largestSect])-1, chi)
			threshold = S[largestSect][chi]
			s_disc = 0
			for s in range(len(S)):
				mask = S[s] > threshold
				s_disc += np.sum(S[s][~mask]**2)
				S[s] = S[s][mask]
				Vh[s] = Vh[s][mask,:]
			trunc = True
	elif totStates > chi:
		if truncation_type == "global":
			S_ToTrunc = [sub_list[chi_block:] for sub_list in S]
			S_ToKeep = [sub_list[:chi_block] for sub_list in S]
			new_totStates = np.sum([len(sub_list) for sub_list in S_ToTrunc])
			if new_totStates > chi: ##Careful: sing. values arrive normed, but leave trunc. not normed.
				sect = np.concatenate([np.full(len(sublist), i) for i, sublist in enumerate(S_ToTrunc)])
				coord = np.concatenate([np.arange(len(sublist)) for sublist in S_ToTrunc])

				# Sort all the eigenvalues and keep only top chi
				order = np.argpartition(np.concatenate(S_ToTrunc), -chi)[-chi:]
				order_trunc = np.argpartition(np.concatenate(S_ToTrunc), -chi)[:-chi]
				sectSort = sect[order]
				sList = np.concatenate(S_ToTrunc)
				SSort = sList[order] 
				sList_truncated = sList[order_trunc] 
				coordSort = coord[order]

				# Count the new number of states in each sector
				split = np.cumsum(np.bincount(sectSort, minlength=len(S_ToTrunc)))

				# Sort back according to sectors
				order2 = np.argsort(sectSort)
				coordKept = coordSort[order2] + chi_block ## All the indices of the truncated values start at chi_block!
				SKept = SSort[order2]

				# Separate back into arrays: each S is ordered by sector and we know how many S are in each sector
				Strunc = np.split(SKept, split[:-1])
				coordtrunc = np.split(coordKept, split[:-1])

				## We add the values which were kept in each block
				Scombined = [np.concatenate((S_ToKeep[i],Strunc[i])) for i in range(len(S))]
				Coordcombined = [np.concatenate((np.arange(len(S_ToKeep[i])),coordtrunc[i])) for i in range(len(S))]

				s_disc = np.sum(sList_truncated**2)

				# Use list comprehension for Vhtrunc
				Vhtrunc = [Vh[s][Coordcombined[s], :] for s in range(len(S))]

				S, Vh = Scombined, Vhtrunc

				trunc = True
				
		elif truncation_type == "block":
			for s in range(len(S)):
				S[s] = S[s][:chi]
				Vh[s] = Vh[s][:chi,:]
			trunc = True
	if trunc==False:
			s_disc = 0
	
	return S, Vh, trunc, s_disc

#---------------------------------------------------------------
def ApplyLambda(Lam, A):
	## Lambda always applied on the LEFT virtual leg (0)
	lA = np.arange(A.nLegs)
	## We multiply Lambda on the left, so A is on the right (summed legs are in the middle)
	LlegsA, RlegsA = np.zeros(1, dtype=int), np.setdiff1d(lA, 0) 

	Sectors = np.intersect1d(Lam.Rsect, A.LegsSectors[0])
	nSectors = len(Sectors)

	A_blocks_mat, A_blocks_shapes = so.reshape_data_Tensors(A, LlegsA, RlegsA)

	input_indices = np.arange(0,A.nSect)

	## We split the sectors according to the LEFT virtual bond dimension (leg: 0)
	Coord_sectors = A.coordinates[0][None,:] == Sectors[:, None] ## Aelems[block, elem-of-A which is in Block]

	## reshape and defines the different blocks
	B = smpo.symmetric_Tensor(A.L, A.d, A.phys_dims, from_others=True, nLegs=A.nLegs, nSect=A.nSect, alpha=A.alpha, data_as_tensors=A.data_as_tensors) ## Defines the new tensor
	B.arrows = A.arrows.copy()
	B.legType = A.legType.copy()
	if A.data_as_tensors == False:
		B.coordinates = A.coordinates.copy()
		B.shapes = A.shapes.copy()

	output_indices = np.zeros(A.nSect, dtype=int) ## To keep the same order of coordinates as before the multiplication

	nCoo = 0
	for s in range(nSectors):
		A_matShapesSect = A_blocks_shapes[:,Coord_sectors[s]]
		nBlocks_sect = np.sum(Coord_sectors[s])
		## Coordinate of the legs wich are not summed, which becomes the new coordinates. One repeat, one tile: 
		if A.data_as_tensors:
			new_coordinates = np.concatenate((np.tile(Lam.Lsect[s], (1,nBlocks_sect)), A.coordinates[RlegsA][:, Coord_sectors[s]]), axis=0)
			B.coordinates[:,nCoo:(nCoo + nBlocks_sect)] = new_coordinates
			## Defines the left and right sizes of each non-zero block of the new tensor
			B.shapes[:,nCoo:(nCoo + nBlocks_sect)] = A.shapes[:,Coord_sectors[s]]

		output_indices[nCoo:(nCoo + nBlocks_sect)] = input_indices[Coord_sectors[s]]

		## Construct the matrix by putting the different blocks together
		# Multiply by s and divide the new matrix in a set of blocks, which are sorted according to the new coordinates B.coordinates
		matA = np.concatenate(A_blocks_mat[Coord_sectors[s,:]], axis=1)
		matB = np.diag(Lam.data[Sectors[s]]) @ matA

		split_matB = np.array_split(matB, np.cumsum(A_matShapesSect[1][:-1]), axis=1)	
		if A.data_as_tensors:
			B.data[nCoo:(nCoo + nBlocks_sect)] = [split_matB[i].astype(complex).reshape(B.shapes[:,nCoo+i]) for i in range(nBlocks_sect)]
		else:
			B.data[nCoo:(nCoo + nBlocks_sect)] = [split_matB[i].astype(complex) for i in range(nBlocks_sect)]

		nCoo += nBlocks_sect ## new number of non-zero sectors defined for the sparse representation C

	B.LegsSectors = np.array(tuple([np.unique(B.coordinates[i,:]) for i in range(B.nLegs)]), dtype=object)

	## Replace the coordinates in the same order as before the multiplication
	if A.data_as_tensors:
		B.coordinates[:,output_indices] = B.coordinates[:,input_indices].copy()
		B.shapes[:,output_indices] = B.shapes[:,input_indices].copy()

	B.data[output_indices] = B.data[input_indices].copy()

	return B

#---------------------------------------------------------------
def Sym_Canonic(PhiB, M, l1, l2, truncate='False', truncation_type="global", last_step = False):
	'''
	Given one tensor A and indices, reshape the tensor to a (set of) matrix(ces).  ## Useless
	'''
	alpha = PhiB.alpha

	Lam = M.TN["Lam%s"%l1]
	## Apply Lambda first: Phi = Lambda * PhiB
	Phi = ApplyLambda(Lam, PhiB) ## The order of blocks of Phi might change from those of PhiB!

	## Reshape the new tensor Phi AND PhiB exactly the same way (they have the same indices! Just the data changes ...)
	lPhi = np.arange(Phi.nLegs)
	Llegs, Rlegs = [0,1,3], [2,4,5]

	Phi_blocks_mat, Phi_blocks_shapes   = so.reshape_data_Tensors(Phi, Llegs, Rlegs)
	PhiB_blocks_mat, PhiB_blocks_shapes = so.reshape_data_Tensors(PhiB, Llegs, Rlegs)
	
	Lmask_arrow, Lmask_s = Phi.arrows[Llegs] == 'i', Phi.legType[Llegs] == 's' ## Incoming legs and direct space legs
	## Direction of the flow
	Lblocks = np.sum(Phi.coordinates[Llegs] * (Lmask_arrow[:,None] - 1*(~Lmask_arrow)[:,None]) * ((Phi.alpha)*Lmask_s[:,None] + (~Lmask_s[:,None])),axis=0) 

	## By definition, the right blocks will be equal to the left blocks
	Sectors = np.unique(Lblocks) ## Defines the DIFFERENT blocks, and how many sub blocks are there in each block (in var. count)
	nSectors = len(Sectors)

	## Mask: tells which coordinates belongs to which MIDDLE block: bool of shape (nBlocks, nCoordinates)
	Coord_sectors = (Lblocks == Sectors[:, None]) ## Blocks_sectors[block, elem-of-A which is in Block]

	BlockCoo, Phi_matShapesSect, nLsect, nRsect, _ = so.construct_subBlocks_Sectors(Phi, Llegs, Rlegs, Phi_blocks_shapes, Coord_sectors, nSectors)

	## Create new tensors
	Lam1 = smpo.symmetric_Lambda(Phi.L, Phi.d, None, initial="from_SVD", chi_max=Lam.chi_max, chi_block=Lam.chi_block)
	B1 = smpo.symmetric_Tensor(Phi.L, Phi.d, np.sum(Phi.legType[Llegs]=='s'), from_others=True, nLegs=len(Llegs)+1, nSect=np.sum(nLsect), alpha=Phi.alpha,
	data_as_tensors=Phi.data_as_tensors) ## Defines the new tensor
	B2 = smpo.symmetric_Tensor(Phi.L, Phi.d, np.sum(Phi.legType[Rlegs]=='s'), from_others=True, nLegs=len(Rlegs)+1, nSect=np.sum(nRsect), alpha=Phi.alpha,
	data_as_tensors=Phi.data_as_tensors) ## Defines the new tensor

	B1.arrows = np.array(tuple(Phi.arrows[Llegs]) + tuple('o',))
	B1.legType = np.array(tuple(Phi.legType[Llegs]) + tuple('v',))
	B2.arrows = np.array(tuple('i',)  + tuple(Phi.arrows[Rlegs]))
	B2.legType = np.array(tuple('v',) + tuple(Phi.legType[Rlegs]))

	whvirt_B1, whvirt_B2 = B1.legType == "v", B2.legType == "v"

	time_svd, number_svds, biggest_svd_cost, summ_cost = 0, 0, 0, 0

	matPhiB, S, Vh = np.zeros(nSectors, dtype=object), np.zeros(nSectors, dtype=object), np.zeros(nSectors, dtype=object)
	block_shape_L, block_shape_R, block_list_L, block_list_R = np.zeros(nSectors, dtype=object), np.zeros(nSectors, dtype=object), np.zeros(nSectors, dtype=object), np.zeros(nSectors, dtype=object)
	norm = 0
	for s in range(nSectors):
		matPhi, block_list_L[s], block_list_R[s], block_shape_L[s], block_shape_R[s] = so.construct_matrix_from_subBlocks(Phi_blocks_mat, Phi_matShapesSect[s], BlockCoo[s],
		Coord_sectors[s])
		matPhiB[s], _,_,_,_ = so.construct_matrix_from_subBlocks(PhiB_blocks_mat, Phi_matShapesSect[s], BlockCoo[s], Coord_sectors[s])

		nn, mm = np.max(matPhi.shape), np.min(matPhi.shape)
		if nn * mm**2 > biggest_svd_cost:
			biggest_svd_cost = nn * mm**2
		summ_cost += nn * mm**2

		try: 
			U, Stemp, Vhtemp = np.linalg.svd(matPhi, full_matrices=False)
		except Exception:
			print("Numpy SVD didn't converge. Using driver 'gesvd'.", "Matrix contained nan: ", np.isnan(np.sum(matPhi)), "Matrix norm: ", np.linalg.norm(matPhi))
			U, Stemp, Vhtemp = sp.linalg.svd(matPhi, full_matrices=False, lapack_driver='gesvd')

		number_svds += 1

		Stemp = Stemp #/ np.linalg.norm(Stemp) ## We normalize the values to 1

		## Threshold of singular values
		mask_th = Stemp > M.th_sing_vals
		S[s] = Stemp[mask_th]
		#print(S[s])
		Vh[s] = Vhtemp[mask_th,:]

		norm += np.sum(S[s]**2)

	norm = np.sqrt(norm)

	totStates, largestS = 0, -1
	## We divide by the overall norm of S (l^2 summed for all sectors)
	for s in range(nSectors):
		Stemp = S[s].copy()
		Stemp = Stemp / norm
		mask_th = Stemp > M.th_sing_vals
		S[s] = Stemp[mask_th]
		Vh[s] = Vh[s][mask_th,:]

		if len(S[s])>0:
			if S[s][0] > largestS:
				largestS = S[s][0]
				largestSect = s

		totStates += len(S[s])

	## After building all the sectors we truncate if required
	S, Vh, trunc, s_disc = truncation(S, Vh, Lam1.chi_max, Lam1.chi_block, truncation_type, totStates, largestSect)

	## From every Vh in every sector, build the corresponding B matrices (Hasting's trick)
	checkB1 = np.ones(np.sum(nLsect), dtype=bool)
	checkB2 = np.ones(np.sum(nRsect), dtype=bool) 
	nCooB1, nCooB2 = 0, 0
	for s in range(nSectors):
		if S[s].shape[0] == 0:
			checkB1[nCooB1:(nCooB1 + nLsect[s])] = False
			checkB2[nCooB2:(nCooB2 + nRsect[s])] = False
		else:
			Lam1.data[Sectors[s]] = S[s]

			B2.coordinates[:,nCooB2:(nCooB2 + nRsect[s])] = np.concatenate((np.tile(np.array((Sectors[s])), (1,nRsect[s])), Phi.coordinates[Rlegs][:,Coord_sectors[s]][:,block_list_R[s]]), axis=0)
			B1.coordinates[:,nCooB1:(nCooB1 + nLsect[s])] = np.concatenate((Phi.coordinates[Llegs][:, Coord_sectors[s]][:,block_list_L[s]], np.tile(np.array((Sectors[s])), (1,nLsect[s]))), axis=0)

			split_matVh = np.array_split(Vh[s], np.cumsum(block_shape_R[s][:-1]), axis=1) ## Split only the columns of Vh 
			matB1 	    = matPhiB[s] @ Vh[s].conj().T
			split_matB1 = np.array_split(matB1, np.cumsum(block_shape_L[s][:-1]), axis=0) ## Split only the rows of B1

			## Check here
			B2.shapes[0,nCooB2:(nCooB2 + nRsect[s])]  = np.tile(np.array((S[s].shape[0])), (1,nRsect[s]))
			B2.shapes[-1,nCooB2:(nCooB2 + nRsect[s])] = block_shape_R[s]
			B1.shapes[0,nCooB1:(nCooB1 + nLsect[s])]  = block_shape_L[s]
			B1.shapes[-1,nCooB1:(nCooB1 + nLsect[s])] = np.tile(np.array((S[s].shape[0])), (1,nLsect[s]))

			if B1.data_as_tensors:
				rsh1 = B1.shapes[:, nCooB1:(nCooB1 + nLsect[s])]
				rsh2 = B2.shapes[:, nCooB2:(nCooB2 + nRsect[s])]
			else:
				rsh1 = B1.shapes[whvirt_B1, nCooB1:(nCooB1 + nLsect[s])]
				rsh2 = B2.shapes[whvirt_B2, nCooB2:(nCooB2 + nRsect[s])]

			B1.data[nCooB1:(nCooB1 + nLsect[s])] = [split_matB1[i].astype(complex).reshape(rsh1[:,i]) for i in range(nLsect[s])]
			B2.data[nCooB2:(nCooB2 + nRsect[s])] = [split_matVh[i].astype(complex).reshape(rsh2[:,i]) for i in range(nRsect[s])]

		nCooB1 += nLsect[s] 
		nCooB2 += nRsect[s] 
	B1 = smpo.MaskCoord(B1, checkB1)
	B2 = smpo.MaskCoord(B2, checkB2)

	Lam1.def_sects(np.array(list(Lam1.data.keys())))

	M.TN["B%s"%l1], M.TN["Lam%s"%l2], M.TN["B%s"%l2] = B1, Lam1, B2

	if trunc:
		M = CheckTruncation(M, l1, l2)

	return M, time_svd, number_svds, summ_cost, s_disc

