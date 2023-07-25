import numpy as np
import scipy.linalg as la
from tqdm import tqdm

from other_functions import *

def get_POD_matrices(Q1, Q2, r, rank=5):
    snaps_Q1 = Q1.T
    U_Q1, Sig_Q1, Vh_Q1 = np.linalg.svd(snaps_Q1, full_matrices=False)
    U_Q1 = U_Q1[:, :rank]

    snaps_Q2 = Q2.T
    U_Q2, Sig_Q2, Vh_Q2 = np.linalg.svd(snaps_Q2, full_matrices=False)
    U_Q2 = U_Q2[:, :rank]

    snaps_r = r.T
    U_r, Sig_r, Vh_r = np.linalg.svd(snaps_r, full_matrices=False)
    U_r = U_r[:, :rank]
    
    return U_Q1, U_Q2, U_r

def initialize_Q_flow_POD(Q1, Q2, p1, p2, r, U_Q1, U_Q2, U_r):
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

    p1_ = np.copy(p1)
    p2_ = np.copy(p2)
    
    r_ = np.zeros((U_r.shape[1], r.shape[0]))
    r_[:, 0] = (U_r.T @ r[0].reshape(-1, 1)).reshape(-1)
    r_ = r_.T
    
    return Q1_, Q2_, p1_, p2_, r_

def solve_Q_flow_linear_system_POD(Q1, Q2, p1_, p2_, r, gamma, U_Q1, U_Q2, LHS, RHS, M):
    """
    returns: Q1[time], Q2[time]
    """

    g1 = U_Q1.T @ (M*(gamma * p1_ * r).reshape(-1, 1))
    g2 = U_Q2.T @ (M*(gamma * p2_ * r).reshape(-1, 1))
    g = RHS @ np.concatenate((Q1.reshape(-1, 1), Q2.reshape(-1, 1)), axis=0) - np.concatenate((g1, g2), axis=0)

    Q_ = np.linalg.solve(LHS, g).reshape(-1)

    return Q_

def solve_Q_flow_POD(Q1, Q2, p1, p2, r, gamma, stiffness_matrix, U_Q1, U_Q2, U_r, Nt, a, b, c, A0, M, L, dt):

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
        Q_ = solve_Q_flow_linear_system_POD(Q1[time-1], Q2[time-1], 
                                            p1[time-1].reshape(-1, 1), p2[time-1].reshape(-1, 1), 
                                            U_r @ r[time-1].reshape(-1, 1), gamma.reshape(-1, 1), U_Q1, U_Q2, LHS, RHS, M)
        Q1[time] = Q_[:Q1_dim]
        Q2[time] = Q_[Q1_dim:]

        # update r
        r[time] = r[time-1] + (U_r.T @ (2 * p1[time-1].reshape(-1, 1) * U_Q1 @ (Q1[time] - Q1[time-1]).reshape(-1, 1)\
            + 2 * p2[time-1].reshape(-1, 1) * U_Q2 @ (Q2[time] - Q2[time-1]).reshape(-1, 1))).reshape(-1)
        
        # update P(Q)
        update_P(time, 
                 (U_Q1 @ Q1[time].reshape(-1, 1)).reshape(-1), 
                 (U_Q2 @ Q2[time].reshape(-1, 1)).reshape(-1), 
                 p1, p2, a, b, c, A0)