import numpy as np
import scipy.linalg as la
from tqdm import tqdm

from other_functions import *

def get_DEIM_operators(nonlinearity, deim_modes=5):

    # Compute SVD of nonlinearity snapshots
    U, Sig, Vh = np.linalg.svd(nonlinearity, full_matrices=False)
    U = U[:, :deim_modes]

    # Do row selection on the left singular vectors
    U_ = np.copy(U)
    indices = []
    for k in range(U.shape[1]):
        i_star = np.argmax(np.abs(U_[:, k]))
        indices.append(i_star)
        U_ = U_ - U_[:, [k]] @ U_[[i_star]] / U_[i_star, k]

    S = np.zeros(U.shape)
    for i, index in enumerate(indices):
        S[index, i] = 1
        
    return U @ np.linalg.inv(S.T @ U), indices

def initialize_Q_flow_DEIM(Q1, Q2, p1, p2, r, U_Q1, U_Q2, U_r, nQ_indices, nR_indices):
    """
    Initialize variables for POD FEM based on
    initial values of variables for vanilla FEM
    """

    Q1_ = np.zeros((U_Q1.shape[1], Q1.shape[0]))
    Q2_ = np.zeros((U_Q2.shape[1], Q2.shape[0]))
    Q1_[:, 0] = (U_Q1.T @ Q1[0].reshape(-1, 1)).reshape(-1)
    Q2_[:, 0] = (U_Q2.T @ Q2[0].reshape(-1, 1)).reshape(-1)
    Q1_ = Q1_.T
    Q2_ = Q2_.T

    p1_Q = np.copy(p1[:, nQ_indices])
    p2_Q = np.copy(p2[:, nQ_indices])

    p1_R = np.copy(p1[:, nR_indices])
    p2_R = np.copy(p2[:, nR_indices])
    
    r_ = np.zeros((U_r.shape[1], r.shape[0]))
    r_[:, 0] = (U_r.T @ r[0].reshape(-1, 1)).reshape(-1)
    r_ = r_.T
    
    return Q1_, Q2_, p1_Q, p2_Q, p1_R, p2_R, r_

def solve_Q_flow_linear_system_DEIM(Q1, Q2, p1_Q, p2_Q, r, gamma, U_r, U_deimQ1, U_deimQ2, nQ_indices, LHS, RHS, M):
    """
    returns: Q1[time], Q2[time]
    """

    g1 = M * U_deimQ1 @ (gamma[nQ_indices] * p1_Q * (U_r[nQ_indices] @ r))
    g2 = M * U_deimQ2 @ (gamma[nQ_indices] * p2_Q * (U_r[nQ_indices] @ r))
    g = RHS @ np.concatenate((Q1.reshape(-1, 1), Q2.reshape(-1, 1)), axis=0) - np.concatenate((g1, g2), axis=0)

    Q_ = np.linalg.solve(LHS, g).reshape(-1)

    return Q_

def solve_Q_flow_DEIM(Q1, Q2, p1_Q, p2_Q, p1_R, p2_R, r, 
                      gamma, stiffness_matrix, 
                      U_Q1, U_Q2, U_r, U_deimQ1, U_deimQ2, U_deimR,
                      nQ_indices, nR_indices,
                      Nt, a, b, c, A0, M, L, dt):

    Q1_dim = Q1.shape[1]

    U1TgammaU1 = U_Q1.T @ np.diag(gamma) @ U_Q1
    U1TstiffU1 = U_Q1.T @ stiffness_matrix @ U_Q1

    U2TgammaU2 = U_Q2.T @ np.diag(gamma) @ U_Q2
    U2TstiffU2 = U_Q2.T @ stiffness_matrix @ U_Q2

    LHS = la.block_diag(U1TgammaU1/dt + M*L/2 * U1TstiffU1, U2TgammaU2/dt + M*L/2 * U2TstiffU2)
    RHS = la.block_diag(U1TgammaU1/dt - M*L/2 * U1TstiffU1, U2TgammaU2/dt - M*L/2 * U2TstiffU2)

    print("Computing scheme...")
    # follow the scheme to solve for the other times
    for time in tqdm(range(1, Nt), position=0, leave=True):

        # update Q1, Q2
        Q_ = solve_Q_flow_linear_system_DEIM(Q1[time-1], Q2[time-1], 
                                             p1_Q[time-1].reshape(-1, 1), p2_Q[time-1].reshape(-1, 1), r[time-1].reshape(-1, 1), 
                                             gamma.reshape(-1, 1), U_r, U_deimQ1, U_deimQ2, nQ_indices,
                                             LHS, RHS, M)
        Q1[time] = Q_[:Q1_dim]
        Q2[time] = Q_[Q1_dim:]

        # update r
        r[time] = r[time-1] + (U_deimR @ (2 * p1_R[time-1].reshape(-1, 1) * U_Q1[nR_indices] @ (Q1[time] - Q1[time-1]).reshape(-1, 1)\
            + 2 * p2_R[time-1].reshape(-1, 1) * U_Q2[nR_indices] @ (Q2[time] - Q2[time-1]).reshape(-1, 1))).reshape(-1)
        
        # update P(Q)
        update_P(time, 
                 (U_Q1[nQ_indices] @ Q1[time].reshape(-1, 1)).reshape(-1), 
                 (U_Q2[nQ_indices] @ Q2[time].reshape(-1, 1)).reshape(-1), 
                 p1_Q, p2_Q, a, b, c, A0)
        
        update_P(time, 
                 (U_Q1[nR_indices] @ Q1[time].reshape(-1, 1)).reshape(-1), 
                 (U_Q2[nR_indices] @ Q2[time].reshape(-1, 1)).reshape(-1), 
                 p1_R, p2_R, a, b, c, A0)