"""
Trotter decomposition module for quantum simulation.

This module implements various orders of Trotter-Suzuki decomposition
for quantum Hamiltonians, optimized for performance and readability.
"""

import numpy as np
import sys
from typing import Dict, List, Tuple, Optional

# Module binding - consider using environment variables or config file
sys.path.append("C:/Users/maxim/Documents/GitHub/Few_body_operators/Codes/Modules/")

import symmpo as smpo
import scipy.linalg as linalg


# Pre-compute common Pauli matrices as constants (computed once)
_SP = np.array([[0, 0], [1., 0]], dtype=np.complex128)
_SZ = np.array([[1., 0.], [0, -1.]], dtype=np.complex128) * 0.5


def commuting_sequence(n_parts: int, dt: float) -> List[Tuple[str, float]]:
    """
    Generate a sequence of commuting Hamiltonian parts.
    
    Args:
        n_parts: Number of commuting parts
        dt: Time step
        
    Returns:
        List of tuples (Hamiltonian_label, timestep)
    """
    return [(f"H{i}", dt) for i in range(n_parts)]


def compute_gate(step: str, V: float, Uint: float, gamma: float, dt: float, 
                 J: float, Jz: float, hamiltonian: str) -> np.ndarray:
    """
    Compute the time evolution gate for a given Hamiltonian.
    
    Args:
        step: Hamiltonian step identifier
        V: Coupling strength parameter
        Uint: On-site interaction parameter
        gamma: Hopping parameter
        dt: Time step
        J: Exchange coupling parameter
        Jz: Ising coupling parameter
        hamiltonian: Model type ('IRLM' or 'Heis_nn')
        
    Returns:
        4D tensor representing the two-site gate
    """
    # Use pre-computed matrices
    Sp, Sz = _SP, _SZ
    
    # Compute Hamiltonian based on model type
    if hamiltonian == "IRLM":
        if step == "H0":
            # Impurity Hamiltonian
            hi = (V * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + 
                  Uint * np.kron(Sz, Sz))
        else:
            # Bath hopping
            hi = gamma * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp))
    elif hamiltonian == "Heis_nn":
        # Heisenberg nearest-neighbor
        hi = (0.5 * J * (np.kron(Sp, Sp.T) + np.kron(Sp.T, Sp)) + 
              Jz * np.kron(Sz, Sz))
    else:
        raise ValueError(f"Unknown Hamiltonian model: {hamiltonian}")
    
    # Compute time evolution operator
    U = linalg.expm(-1j * hi * dt).reshape(2, 2, 2, 2)
    return U


def _create_symmetric_gates(step: str, dt_values: List[float], phys_dims: int,
                            params: Dict, model: str, alpha: float,
                            data_as_tensors: bool) -> Tuple[Dict, Dict]:
    """
    Helper function to create symmetric gates for multiple time steps.
    
    Returns:
        Tuple of (U_dict, Udag_dict)
    """
    U, Udag = {}, {}
    for dt in dt_values:
        U[step, dt] = smpo.symmetric_Gate(
            2 * phys_dims, dt, params, 
            gateType="Hamiltonian", model=model, alpha=alpha,
            data_as_tensors=data_as_tensors, step=step
        )
        Udag[step, dt] = smpo.symmetric_Gate(
            2 * phys_dims, dt, params,
            gateType="Hamiltonian", model=model, dag=True, alpha=alpha,
            data_as_tensors=data_as_tensors, step=step
        )
    return U, Udag


def _create_nonsymmetric_gates(step: str, dt_values: List[float], 
                               V: float, Uint: float, gamma: float,
                               J: float, Jz: float, model: str) -> Dict:
    """
    Helper function to create non-symmetric gates for multiple time steps.
    
    Returns:
        Dictionary of gates
    """
    return {(step, dt): compute_gate(step, V, Uint, gamma, dt, J, Jz, model)
            for dt in dt_values}


def _compact_consecutive_steps(steps: List[Tuple]) -> List[Tuple]:
    """
    Merge consecutive steps with the same Hamiltonian part.
    
    Args:
        steps: List of (Hamiltonian_label, timestep) tuples
        
    Returns:
        Compacted list with merged consecutive identical steps
    """
    if not steps:
        return []
    
    compact_steps = []
    i = 0
    
    while i < len(steps) - 1:
        if steps[i][0] == steps[i + 1][0]:
            # Merge consecutive identical steps
            compact_steps.append((steps[i][0], steps[i][1] + steps[i + 1][1]))
            i += 2
        else:
            compact_steps.append(steps[i])
            i += 1
    
    # Add last step if not already processed
    if i == len(steps) - 1:
        compact_steps.append(steps[-1])
    
    return compact_steps


def trotter_sequence(trotter_order: int, n_steps: int, dt: float, n_parts: int,
                     n_steps_calc: int, params: Dict, t: float = 0,
                     period_t: float = 0, h_mean: float = 0, h_drive: float = 0,
                     alpha: float = -1, data_as_tensors: bool = True,
                     model: str = "Heis_nn", h_steps: List[str] = None,
                     symm: bool = False) -> Tuple[List, Dict, Dict, int]:
    """
    Generate Trotter-Suzuki decomposition sequence.
    
    Args:
        trotter_order: Order of Trotter decomposition (1, 2, or 4)
        n_steps: Number of time steps
        dt: Time step size
        n_parts: Number of commuting Hamiltonian parts
        n_steps_calc: Calculate observables every n_steps_calc steps
        params: Dictionary of parameters
        symm: Use symmetric gates if True
        model: Hamiltonian model type
        h_steps: List of Hamiltonian step identifiers
        
    Returns:
        Tuple of (compact_steps, U_gates, Udag_gates, n_compact_steps)
    """
    if h_steps is None:
        h_steps = ['H0', 'H1']
    
    U, Udag = {}, {}
    
    # Extract parameters from dictionary
    globals().update(params)
    
    # Generate gate sequence based on Trotter order
    if trotter_order == 1:
        # First order: O(dt^2) error
        for step in h_steps:
            if symm:
                phys_dims = params.get('phys_dims', 2)
                u_dict, udag_dict = _create_symmetric_gates(
                    step, [dt], phys_dims, params, model, alpha, data_as_tensors
                )
                U.update(u_dict)
                Udag.update(udag_dict)
            else:
                U[step, dt] = compute_gate(
                    step, V, Uint, gamma, dt, J, Jz, model
                )
        
        steps = commuting_sequence(n_parts, dt)
    
    elif trotter_order == 2:
        # Second order: O(dt^3) error
        # Symmetric decomposition: U(dt/2) U(dt) U(dt/2)
        dt_list = [dt, dt / 2]
        
        for step in h_steps:
            if symm:
                phys_dims = params.get('phys_dims', 2)
                u_dict, udag_dict = _create_symmetric_gates(
                    step, dt_list, phys_dims, params, model, alpha, data_as_tensors
                )
                U.update(u_dict)
                Udag.update(udag_dict)
            else:
                for dt_val in dt_list:
                    U[step, dt_val] = compute_gate(
                        step, V, Uint, gamma, dt_val, J, Jz, model
                    )
        
        seq_half = commuting_sequence(n_parts, dt / 2)
        steps = seq_half + seq_half[::-1]
    
    elif trotter_order == 4:
        # Fourth order: O(dt^5) error
        # Yoshida's coefficients
        factor = 4.0 ** (1 / 3.0)
        dt1 = dt / (4.0 - factor)
        dt2 = dt - 4.0 * dt1
        
        # Build second-order sequences
        seq1_half = commuting_sequence(n_parts, dt1 / 2)
        u2_1 = seq1_half + seq1_half[::-1]
        
        seq2_half = commuting_sequence(n_parts, dt2 / 2)
        u2_2 = seq2_half + seq2_half[::-1]
        
        dt_list = [dt1, dt2, dt1 / 2, dt2 / 2, dt1 + dt2, (dt1 + dt2) / 2]
        
        for step in h_steps:
            if symm:
                phys_dims = params.get('phys_dims', 2)
                u_dict, udag_dict = _create_symmetric_gates(
                    step, dt_list, phys_dims, params, model, alpha, data_as_tensors
                )
                U.update(u_dict)
                Udag.update(udag_dict)
            else:
                for dt_val in dt_list:
                    U[step, dt_val] = compute_gate(
                        step, V, Uint, gamma, dt_val, J, Jz, model
                    )
        
        # Fourth-order sequence: U2(dt1) U2(dt1) U2(dt2) U2(dt1) U2(dt1)
        steps = u2_1 + u2_1 + u2_2 + u2_1 + u2_1
    
    else:
        raise ValueError(f"Trotter order {trotter_order} not supported. Use 1, 2, or 4.")
    
    # First compaction: merge consecutive identical gates
    compact_steps = _compact_consecutive_steps(steps)
    n_compact_steps = len(compact_steps)
    
    # Replicate for multiple time steps
    steps = compact_steps * n_steps
    
    # Second compaction: add time tracking and observable calculation flags
    # CRITICAL: Steps must NOT be merged when observable calculation is required
    # This preserves the exact time point where measurements should be taken
    compact_steps = []
    i, time_counter = 0, 0
    
    while i < len(steps) - 1:
        calc_obs = False
        new_time = False
        
        # Determine current period (which complete Trotter step we're in)
        current_period = i // n_compact_steps
        
        # Check how many steps could potentially be merged
        n_combine = 2 if steps[i][0] == steps[i + 1][0] else 1
        
        # Check if merging would cross into a new time period
        if (i + n_combine) // n_compact_steps != current_period:
            new_time = True
            time_counter += 1
            
            # Mark for observable calculation if at the right interval
            if time_counter % n_steps_calc == 0:
                calc_obs = True
        
        # MERGE DECISION: Only merge if same Hamiltonian AND no observable calculation
        # The check for calc_obs ensures we don't compress across measurement points
        can_merge = steps[i][0] == steps[i + 1][0] and not calc_obs
        
        if can_merge:
            compact_steps.append((
                steps[i][0],
                steps[i][1] + steps[i + 1][1],
                new_time,
                calc_obs  # Will be False here due to merge condition
            ))
            i += 2
        else:
            compact_steps.append(steps[i] + (new_time, calc_obs))
            i += 1
    
    # Handle final step (no more steps to merge with)
    if i == len(steps) - 1:
        compact_steps.append(steps[-1] + (new_time, calc_obs))
    
    return compact_steps, U, Udag, n_compact_steps


# Backward compatibility aliases
Commuting_Seq = commuting_sequence
gates = compute_gate
Trotter_Seq = trotter_sequence
