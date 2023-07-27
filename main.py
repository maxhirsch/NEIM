import numpy as np
import torch
import torch.nn as nn
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm

from vanilla_fem import *
from pod_fem import *
from deim_fem import *
from neim_fem import *
from other_functions import *


PERFORM_DEIM = False

# Omega = [0, Lx] x [0, Lx]
# Lx is currently hard coded as 2 in mesh generation, 
# so changing this does nothing.
Lx = 2 

# model parameters
a = -0.2
b = 1
c = 1
A0 = 500
M = 1
L1 = 0.1

# number of time steps and final time
Nt = 80
t_final = 4
tVals = np.linspace(0, t_final, Nt)
dt = tVals[1] - tVals[0] # delta t


# define the mesh in space
mesh, interior_point_coords = get_triangulation(refinements=5)

# Get vectors for the components of matrices (flattened over space) 
# over time with time 0 initialized.
# We only keep track of nodes in the interior of \Omega because we
# use homogeneous Dirichlet boundary conditions.
Q1, Q2, p1, p2, r, num_interior_points = initialize_Q_flow(interior_point_coords, Nt, a, b, c, A0)

# Get the finite element space, stiffness matrix, lumped mass matrix as a vector, and mass matrix
Vh, stiffness_matrix, gamma, mass_matrix = calculate_basis_integrals(mesh)

# solve the PDE (updates the entries of Q1, Q2, etc. within the function)
solve_Q_flow(Q1, Q2, p1, p2, r, gamma, stiffness_matrix, Nt, a, b, c, A0, M, L1, dt)

visualize(interior_point_coords, Q1, Q2, r, num_interior_points, Nt, every=1)

rank = 5

snaps_Q1 = np.copy(Q1).T
U_Q1, Sig_Q1, Vh_Q1 = np.linalg.svd(snaps_Q1, full_matrices=False)
U_Q1 = U_Q1[:, :rank]
print(U_Q1.shape, Sig_Q1.shape, Vh_Q1.shape)

snaps_Q2 = np.copy(Q2).T
U_Q2, Sig_Q2, Vh_Q2 = np.linalg.svd(snaps_Q2, full_matrices=False)
U_Q2 = U_Q2[:, :rank]
print(U_Q2.shape, Sig_Q2.shape, Vh_Q2.shape)

snaps_r = np.copy(r).T
U_r, Sig_r, Vh_r = np.linalg.svd(snaps_r, full_matrices=False)
U_r = U_r[:, :rank]
print(U_r.shape, Sig_r.shape, Vh_r.shape)





Q1_pod, Q2_pod, p1_pod, p2_pod, r_pod = initialize_Q_flow_POD(Q1, Q2, p1, p2, r, U_Q1, U_Q2, U_r)
solve_Q_flow_POD(Q1_pod, Q2_pod, p1_pod, p2_pod, r_pod, gamma, stiffness_matrix, U_Q1, U_Q2, U_r, Nt, a, b, c, A0, M, L1, dt)

print(Q1, (U_Q1 @ Q1_pod.T).T)


if PERFORM_DEIM:
    """
    DEIM
    """
    # DEIM for Q equation nonlinearity
    deim_modes = 7
    # 1. Compute the nonlinearity at the snapshots
    nonlinearityQ = M * gamma.reshape(-1, 1) * np.concatenate([p1 * r, p2 * r], axis=0).T

    # 2. Compute SVD of nonlinearity snapshots
    U_nQ, Sig_nQ, Vh_nQ = np.linalg.svd(nonlinearityQ, full_matrices=False)
    U_nQ = U_nQ[:, :deim_modes]

    # 3. Do row selection on the left singular vectors
    U_nQ_ = np.copy(U_nQ)
    nQ_indices = []
    for k in range(U_nQ.shape[1]):
        i_star = np.argmax(np.abs(U_nQ_[:, k]))
        nQ_indices.append(i_star)
        U_nQ_ = U_nQ_ - U_nQ_[:, [k]] @ U_nQ_[[i_star]] / U_nQ_[i_star, k]

    S_deimQ = np.zeros(U_nQ.shape)
    for i, index in enumerate(nQ_indices):
        S_deimQ[index, i] = 1
    U_deimQ1 = U_Q1.T @ U_nQ @ np.linalg.inv(S_deimQ.T @ U_nQ)
    U_deimQ2 = U_Q2.T @ U_nQ @ np.linalg.inv(S_deimQ.T @ U_nQ)

    # DEIM for r equation nonlinearity
    # 1. Compute the nonlinearity at the snapshots
    nonlinearityR = (2 * p1[:-1] * (Q1[1:] - Q1[:-1]) + 2 * p2[:-1] * (Q2[1:] - Q2[:-1])).T

    # 2. Compute SVD of nonlinearity snapshots
    U_nR, Sig_nR, Vh_nR = np.linalg.svd(nonlinearityR, full_matrices=False)
    U_nR = U_nR[:, :deim_modes]

    # 3. Do row selection on the left singular vectors
    U_nR_ = np.copy(U_nR)
    nR_indices = []
    for k in range(U_nR.shape[1]):
        i_star = np.argmax(np.abs(U_nR_[:, k]))
        nR_indices.append(i_star)
        U_nR_ = U_nR_ - U_nR_[:, [k]] @ U_nR_[[i_star]] / U_nR_[i_star, k]

    S_deimR = np.zeros(U_nR.shape)
    for i, index in enumerate(nR_indices):
        S_deimR[index, i] = 1
    U_deimR = U_r.T @ U_nR @ np.linalg.inv(S_deimR.T @ U_nR)




    Q1_deim, Q2_deim, p1_deimQ, p2_deimQ, p1_deimR, p2_deimR, r_deim = initialize_Q_flow_DEIM(Q1, Q2, p1, p2, r, U_Q1, U_Q2, U_r, nQ_indices, nR_indices)
    solve_Q_flow_DEIM(Q1_deim, Q2_deim, p1_deimQ, p2_deimQ, p1_deimR, p2_deimR, r_deim, 
                        gamma, stiffness_matrix, 
                        U_Q1, U_Q2, U_r, U_deimQ1, U_deimQ2, U_deimR,
                        nQ_indices, nR_indices,
                        Nt, a, b, c, A0, M, L1, dt)


    print(Q1, (U_Q1 @ Q1_pod.T).T, (U_Q1 @ Q1_deim.T).T)
    print(np.linalg.norm(Q1 - (U_Q1 @ Q1_pod.T).T))
    print(np.linalg.norm(Q1 - (U_Q1 @ Q1_deim.T).T))
    print(np.linalg.norm((U_Q1 @ Q1_pod.T).T - (U_Q1 @ Q1_deim.T).T))


mu = np.array([[a, b, c]])
f_NEIM = M * np.concatenate([U_Q1.T @ (gamma.reshape(1, -1) * p1 * r).T, U_Q2.T @ (gamma.reshape(1, -1) * p2 * r).T], axis=0) # (2dim, times)
f_NEIM = nonlinearityQ[None, :, :]
ro_sols = np.concatenate([U_Q1.T @ Q1.T, U_Q2.T @ Q2.T, U_r.T @ r.T], axis=0).T
print(mu.shape, nonlinearityQ.shape)