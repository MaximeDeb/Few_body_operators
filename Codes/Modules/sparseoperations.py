import numpy as np

#---------------------------------------------------------------
def block_split(array, cuts_rows, cuts_cols):
	'''
	Split a matrix into sub-matrices according to horizontal (rows) and vertical (cols) cuts.
	'''
	split_rows = np.split(array, cuts_rows, axis=0)
	sub_matrices = []
	for row_block in split_rows:
		# Split each row block along columns
		sub_matrices.extend(np.split(row_block, cuts_cols, axis=1))
	return sub_matrices

#---------------------------------------------------------------
def reshape_data_Tensors(A, Llegs, Rlegs):
	'''
	Data reshaped as the corresponding matrix (useful if virtual dimensions are on the same side: |a>|b>, useless if |a><b|)
	'''
	lA = np.arange(A.nLegs)
	matrices = np.zeros(A.nSect, dtype=object)
	if A.data_as_tensors: ## Each tensor is reshaped as a matrix according to legs.
		matrices[:] = [np.moveaxis(dat, Llegs, lA[:len(Llegs)]).reshape(-1, np.prod(A.shapes[Rlegs,i])) for i, dat in enumerate(A.data)]
		shapes = np.array([matrices[i].shape for i in range(A.nSect)]).T

	else: ## Each tensor is already a matrix, which can be: vectorized, transposed, both.
		maskv = A.legType == "v"
		if np.sum(A.legType[Llegs] == 'v') == 1: ## Then it's a matrix, we do nothing
			maskL, maskR = np.sum(lA[:,None] == np.array(Llegs)[None,:], axis=1).astype(bool), np.sum(lA[:,None] == np.array(Rlegs)[None,:], axis=1).astype(bool)
			left, right = lA[maskL * maskv], lA[maskR * maskv]
			if left[0] < right[0]: ## Stored in the usual way
				matrices = A.data
				shapes = A.shapes[maskv]
			else: ## We do the transpose
				matrices[:] = [mat.T for mat in A.data]
				shapes = A.shapes[maskv][::-1,:]
		else: ## Both on the same side
			shapes = np.ones((2,A.nSect), dtype=np.uint32)
			if np.sum(A.legType[Llegs] == 'v') != 0:
				matrices[:] = [mat.reshape(-1,1) for mat in A.data]
				shapes[0] = np.prod(A.shapes[maskv], axis=0)
			else:
				matrices[:] = [mat.reshape(1,-1) for mat in A.data]
				shapes[1] = np.prod(A.shapes[maskv], axis=0)
	return matrices, shapes

#---------------------------------------------------------------
def construct_subBlocks_Sectors(A, Llegs, Rlegs, A_blocks_shapes, Coord_sectors, nSectors):
	'''
	Returns for each block WHICH coordinates of tensor A to put together and WHERE in the final matrix, and the corresponding number of subBlocks, dimensions, etc.
	BlockCoo: if both dimensions have more than one subBlock
	Phi_matShapesSect: Shape of each subBlock
	nLsect, nRsect: number of subBlocks on the left and right of the BIG matrix
	'''
	BlockCoo = np.zeros(nSectors, dtype=object)
	blocksSect = np.zeros((nSectors,2), dtype=object)
	A_matShapesSect = np.zeros(nSectors, dtype=object)
	nLsect, nRsect = np.zeros(nSectors, dtype=int), np.zeros(nSectors, dtype=int)

	Llind, Rlind = np.arange(0,len(Llegs), dtype=int), np.arange(0,len(Rlegs), dtype=int)
	powL = np.power((A.L+1)**2, Llind)
	powR = np.power((A.L+1)**2, Rlind)
	for s in range(nSectors):
		## LEFT - blocks
		Llegs_sect = A.coordinates[Llegs][:, Coord_sectors[s]] ## Legs on the left 
		## Identification number for the sub-blocks coordinates: this one leads to wrong differentiations (same ID for two diff sectors)
		LsubSector_ID = np.sum(powL[:,None] * Llegs_sect[:,:], axis=0) ## Careful with this one 
		Ldiff_sect = np.unique(LsubSector_ID)
		mask_LsubSector = (LsubSector_ID[None,:] == Ldiff_sect[:,None]) ## Who is in which L-block (in A coo.)

		## RIGHT - blocks
		Rlegs_sect = A.coordinates[Rlegs][:, Coord_sectors[s]] ## Legs on the right 
		RsubSector_ID = np.sum(powR[:,None] * Rlegs_sect, axis=0) ## Careful with this one
		Rdiff_sect = np.unique(RsubSector_ID)
		mask_RsubSector = (RsubSector_ID[None,:] == Rdiff_sect[:,None]) ## Who is in which R-block (in B coo.)

		blocksSect[s,0] = Ldiff_sect
		blocksSect[s,1] = Rdiff_sect

		## Shapes (nb of Left, Mid, and Right blocks)
		nLeft, nRight  = len(Ldiff_sect), len(Rdiff_sect) 
		nLsect[s], nRsect[s] = nLeft, nRight

		BlockCoo[s] = np.ones((nLeft, nRight), dtype=int) * -1

		## Left-Right sub-sectors - coordinates
		LR = np.where(mask_LsubSector[:,None,:] * mask_RsubSector[None,:,:])
		BlockCoo[s][LR[0], LR[1]] = LR[2]

		## Shapes of each block
		A_matShapesSect[s] = np.zeros((2, nLeft, nRight), dtype=int)
		## /!\ We can avoid the np.tile, we need only the shapes on the left and middle edges
		A_matShapesSect[s][:, LR[0], LR[1]] = A_blocks_shapes[:, Coord_sectors[s]][:, LR[2]]
		A_matShapesSect[s][0] = A_matShapesSect[s][0].max(axis=1)[:,None]
		A_matShapesSect[s][1] = A_matShapesSect[s][-1].max(axis=0)[None,:]

	return BlockCoo, A_matShapesSect, nLsect, nRsect, blocksSect

#---------------------------------------------------------------
def construct_matrix_from_subBlocks(A_blocks_mat, A_matShapesSect_s, BlockCoo_s, Coord_sectors_s):
	'''
	mat:
	block_list_L:
	block_list_R:
	block_shape_L:
	block_shape_R:
	'''
	## Matrix
	block_list_L = []
	rows = []
	for i, Row_i in enumerate(BlockCoo_s):
		row_list = []
		block_list_L.append(Row_i[Row_i!=(-1)][0])
		for j, Block_ij in enumerate(Row_i):
			if Block_ij != -1: ## Non-empty block
				row_list.append(A_blocks_mat[Coord_sectors_s][Block_ij])
			else: ## Empty block: we fill with 0 of the correct size
				row_list.append(np.zeros(A_matShapesSect_s[:,i,j]))
		rows.append(np.concatenate(row_list, axis=1) )
	mat = np.concatenate(rows, axis=0)
	
	## Store block information for right side
	block_list_R = [col[col!=(-1)][0] for col in BlockCoo_s.T]
	block_shape_L = A_matShapesSect_s[0,:,0].copy()
	block_shape_R = A_matShapesSect_s[1,0,:].copy()

	return mat, block_list_L, block_list_R, block_shape_L, block_shape_R

#---------------------------------------------------------------
def Check_subBlocks_Sectors(L_BlockCoo, R_BlockCoo, A_matShapesSect, B_matShapesSect, IdBlocksA, IdBlocksB):
	'''
	'''
	emptySect = np.zeros(len(L_BlockCoo), dtype=bool)
	for s in range(len(L_BlockCoo)):
		IdMid_L = IdBlocksA[s,1]
		IdMid_R = IdBlocksB[s,0]
		common_Id, indL, indR = np.intersect1d(IdMid_L, IdMid_R, return_indices=True)
		if (common_Id.shape[0] == 0):
			emptySect[s] = True
		L_BlockCoo[s] = L_BlockCoo[s][:, indL]
		R_BlockCoo[s] = R_BlockCoo[s][indR,:]
		A_matShapesSect[s] = A_matShapesSect[s][:,:,indL]
		B_matShapesSect[s] = B_matShapesSect[s][:,indR,:]

	return L_BlockCoo, R_BlockCoo, A_matShapesSect, B_matShapesSect, emptySect

#---------------------------------------------------------------
