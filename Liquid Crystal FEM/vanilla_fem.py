import numpy as np
import matplotlib.pyplot as plt
import skfem as fem
from skfem.helpers import dot, grad
from scipy.sparse import spdiags
from scipy.sparse import kron
from scipy.sparse import linalg
from scipy.sparse import block_diag
from scipy.sparse import eye
from scipy import sparse
from tqdm import tqdm

from pod_fem import *
from deim_fem import *
from other_functions import *

def solve_Q_flow_linear_system(time, Q1, Q2, p1_, p2_, r, gamma, stiffness_matrix, M, L, dt):
    """
    returns: Q1[time], Q2[time]
    """

    num_interior_points = gamma.shape[0]

    gamma_over_dt = 1/dt * spdiags([gamma], [0], num_interior_points, num_interior_points)

    T1 = gamma_over_dt + M*L/2 * stiffness_matrix
    T2 = gamma_over_dt + M*L/2 * stiffness_matrix

    A = block_diag((T1, T2), format = 'csr')

    g1 = (gamma_over_dt - M*L/2 * stiffness_matrix) @ Q1[time-1].reshape(-1,1) - M * (gamma * p1_ * r[time-1]).reshape(-1, 1)
    g2 = (gamma_over_dt - M*L/2 * stiffness_matrix) @ Q2[time-1].reshape(-1,1) - M * (gamma * p2_ * r[time-1]).reshape(-1, 1)
    g = np.concatenate((g1, g2), axis=0)

    Q_ = linalg.spsolve(A, g).reshape(-1)

    return Q_[:num_interior_points], Q_[num_interior_points:]

def solve_Q_flow(Q1, Q2, p1, p2, r, gamma, stiffness_matrix, Nt, a, b, c, A0, M, L, dt):
    """
    solve_Q_flow: 
    
    returns: (xVals, yVals, tVals, Q1, Q2) where
        xVals (np.array) with xVals.shape == (Nx,)
        yVals (np.array) with yVals.shape == (Nx,)
        tVals (np.array) with tVals.shape == (Nt,)
        Q1 (np.array) with Q1.shape == (Nt, Nx**2)
        Q2 (np.array) with Q2.shape == (Nt, Nx**2)
    """

    print("Computing scheme...")
    # follow the scheme to solve for the other times
    for time in tqdm(range(1, Nt), position=0, leave=True):
        # \tilde{p}^{n+1/2}
        p1_ = p1[time-1]
        p2_ = p2[time-1]

        # update Q1, Q2
        Q1_, Q2_ = solve_Q_flow_linear_system(time, Q1, Q2, p1_, p2_, r, gamma, stiffness_matrix, M, L, dt)
        Q1[time] = Q1_
        Q2[time] = Q2_

        # update r
        r[time] = r[time-1] + 2 * p1_ * (Q1[time] - Q1[time-1]) + 2 * p2_ * (Q2[time] - Q2[time-1])
        
        # update P(Q)
        update_P(time, Q1[time], Q2[time], p1, p2, a, b, c, A0)

