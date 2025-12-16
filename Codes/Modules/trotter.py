import numpy as np
import sys

sys.path.append("C:/Users/maxim/Documents/GitHub/Few_body_operators/Codes/Modules/") ## Important to bind the modules

import symmpo as smpo
import scipy as sp

##      ----------------------------------------------------------------------------------------
def Commuting_Seq(nCommParts, dt):
    seq = [("H%s"%i,dt) for i in range(0,nCommParts)]
    return seq

##      ----------------------------------------------------------------------------------------

def gates(step, V, Uint, gamma, dt, J, Jz, hamil):
    Sp =  np.array(((0,0),(1.,0)))
    Sz =  np.array(((1.,0.),(0,-1.))) * 0.5

    if hamil == "IRLM":
        if step == "H0": ## H0 is defined as the gates acting on the impurity, different from the tight-binding in the bath
            hi = V * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + Uint * np.kron(Sz, Sz) 
        else:
            hi = gamma * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
    elif hamil == "Heis_nn":
        hi = 0.5 * J * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + Jz * np.kron(Sz, Sz) 
        
    U = sp.linalg.expm(-1.j * hi * dt).reshape(2,2,2,2)

    return U

##      ----------------------------------------------------------------------------------------
def Trotter_Seq(TrottOrder, Nsteps, dt, nParts, nStepsCalc, Params, t=0, periodT=0, hmean=0, hdrive=0, alpha=-1, data_as_tensors = True, model="Heis_nn", Hsteps=['H0', 'H1'], symm=False):
    U = {}
    Udag = {}
    globals().update(Params)

    ## First order Trotter decomposition: O(t^2)
    if (TrottOrder == 1):
        ## Define the series of gates 'H..' with timestep 'dt', U1 is the brickwall circuit.
        for step in Hsteps:
            if symm:
               U[step,dt]= smpo.symmetric_Gate(2*phys_dims, dt, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
               Udag[step,dt] = smpo.symmetric_Gate(2*phys_dims, dt, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha, data_as_tensors=data_as_tensors,
            step=step)
            else:
                U[step, dt] = gates(step, V, Uint, gamma, dt, J, Jz, model)

        #steps = Commuting_Seq(nParts, dt) * Nsteps

        steps = Commuting_Seq(nParts, dt) #* Nsteps

    ### Second order Trotter decomposition: O(t^3)
    elif (TrottOrder == 2):
        ## Single time step: U2 = U1(t1/2)U1(t1)U1(t1/2)
        dt_list = [dt, dt/2]
        for step in Hsteps:
            for DT in dt_list:
                if symm:
                    U[step,DT]    = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
                    Udag[step,DT] = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha, data_as_tensors=data_as_tensors,step = step)
                else:
                    U[step, DT] = gates(step, V, Uint, gamma, DT, J, Jz, model)

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
                if symm:
                    U[step,DT]    = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
                    Udag[step,DT] = smpo.symmetric_Gate(2*phys_dims, DT, Params, gateType="Hamiltonian", model=model, dag=True, alpha=alpha, data_as_tensors=data_as_tensors, step=step)
                else:
                    U[step, DT] = gates(step, V, Uint, gamma, DT, J, Jz, model)

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
