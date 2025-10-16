import numpy as np
import symmpo as smpo

##      ----------------------------------------------------------------------------------------
def Commuting_Seq(nCommParts, dt):
	seq = [("H%s"%i,dt) for i in range(0,nCommParts)]
	return seq

##      ----------------------------------------------------------------------------------------
def Trotter_Seq(TrottOrder, Nsteps, dt, nParts, nStepsCalc, Params, t=0, periodT=0, hmean=0, hdrive=0, alpha=-1, data_as_tensors = True, model="Heis_nn", Hsteps=['H0', 'H1']):
	U = {}
	Udag = {}
	globals().update(Params)

	## First order Trotter decomposition: O(t^2)
	if (TrottOrder == 1):
		## Define the series of gates 'H..' with timestep 'dt', U1 is the brickwall circuit.
		for step in Hsteps:
			U[step,dt]    = smpo.symmetric_Gate(2*phys_dims, dt, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
			Udag[step,dt] = smpo.symmetric_Gate(2*phys_dims, dt, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha, data_as_tensors=data_as_tensors,
			step=step)

		#steps = Commuting_Seq(nParts, dt) * Nsteps

		steps = Commuting_Seq(nParts, dt) #* Nsteps

	### Second order Trotter decomposition: O(t^3)
	elif (TrottOrder == 2):
		## Single time step: U2 = U1(t1/2)U1(t1)U1(t1/2)
		dt_list = [dt, dt/2]
		for step in Hsteps:
			for DT in dt_list:
				U[step,DT]    = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
				Udag[step,DT] = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha,
				data_as_tensors=data_as_tensors,step = step)

		#steps = (Commuting_Seq(nParts, dt/2) + Commuting_Seq(nParts, dt/2)[::-1]) * Nsteps

		steps = (Commuting_Seq(nParts, dt/2) + Commuting_Seq(nParts, dt/2)[::-1]) #* Nsteps

	### Fourth order Trotter decomposition: O(t^5)
	elif (TrottOrder == 4):
		## 4th order Trotter time steps
		dt1 = dt / (4. - 4.**(1/3.))
		dt2 = (dt - 4. * dt1)
		U2_1 = (Commuting_Seq(nParts, dt1/2) + Commuting_Seq(nParts, dt1/2)[::-1])
		U2_2 = (Commuting_Seq(nParts, dt2/2) + Commuting_Seq(nParts, dt2/2)[::-1])

		dt_list = [dt1, dt2, dt1/2, dt2/2, dt1+dt2, (dt1+dt2)/2]
		for step in Hsteps:
			for DT in dt_list:
				U[step,DT]    = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors,
				step=step)
				Udag[step,DT] = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha,
				data_as_tensors=data_as_tensors, step=step)

		## Single time step: U4 = U2(t1)U2(t1)U2(t2)U2(t1)U2(t1)
		#steps = (U2_1 + U2_1 + U2_2 + U2_1 + U2_1) * Nsteps

		steps = (U2_1 + U2_1 + U2_2 + U2_1 + U2_1)

	### Reduce the effective number of gates that are applied: each similar consecutive sequence of gates is absorbed in a single sequence with the two times summed.
	compactSteps, i = [], 0
	while i < (len(steps)-1):
		if steps[i][0] == steps[i+1][0]:
			compactSteps += [(steps[i][0], steps[i][1]+steps[i+1][1])]
			i += 2
		else:
			compactSteps += (steps[i],)
			i += 1
	compactSteps += (steps[-1],)
	n_comp_Steps = len(compactSteps)

	steps = compactSteps * Nsteps
	compactSteps, i, tt = [], 0, 0
	while i < (len(steps)-1):
		calcObs = False
		newT = False

		a = i//n_comp_Steps
		ncomb = 2 if (steps[i][0] == steps[i+1][0]) else 1
		if ((i+ncomb)//n_comp_Steps != a):
			newT = True
			tt += 1
		if ((tt % nStepsCalc) == 0) and (newT == True):
			calcObs = True

		if (steps[i][0] == steps[i+1][0]) and (calcObs == False):
			compactSteps += [(steps[i][0], steps[i][1]+steps[i+1][1], newT, calcObs)]
			i += 2
		else:
			compactSteps += [steps[i] + (newT, calcObs)]
			i+=1
	compactSteps += [steps[-1] + (newT, calcObs)]

	return compactSteps, U, Udag, n_comp_Steps
