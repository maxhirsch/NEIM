import numpy as np
import matplotlib.pyplot as plt
import skfem as fem
from skfem.helpers import dot, grad
from scipy.sparse import spdiags
from scipy.sparse import kron
from scipy.sparse import linalg
from scipy.sparse import bmat
from scipy.sparse import eye
from scipy import sparse
from tqdm import tqdm


"""
Functions for mesh generation and assembly
"""
def get_triangulation(refinements=5):
    """
    refinements (int): more refinements = finer mesh
    """
    # \Omega = [0,2] x [0,2]
    mesh = fem.MeshTri(doflocs = np.array([[0., 2., 0., 2.], [0., 0., 2., 2.]])).refined(refinements)
    interior_point_coords = np.array(mesh.to_dict()['p'])[mesh.interior_nodes()]
    return mesh, interior_point_coords

def calculate_basis_integrals(mesh):
    """
    Returns a vector with gammas and stiffness matrix for
    interior points of the mesh

    mesh: calculated by get_triangulation
    """
    
    
    # Bilinear form is \int grad(u) * grad(v) dx
    # (For the stiffness matrix)
    @fem.BilinearForm
    def A(u, v, _):
        return dot(grad(u), grad(v))
    
    # Bilinear form is \int u * v dx
    # (For the mass matrix, not needed for scheme,
    #  but needed for POD)
    @fem.BilinearForm
    def M(u, v, _):
        return u * v

    # Linear form is \int v dx
    @fem.LinearForm
    def L(v, w):
        return v

    # define piecewise linear basis functions from the mesh
    Vh = fem.Basis(mesh, fem.ElementTriP1())


    # calculate the stiffness matrix and gamma for all points
    # (including the boundary)
    stiffness_matrix = A.assemble(Vh)
    gamma = L.assemble(Vh)
    mass_matrix = M.assemble(Vh)
    
    # delete boundary degrees of freedom
    stiffness_matrix, _, _, _ = fem.condense(stiffness_matrix, gamma, D=mesh.boundary_nodes())
    mass_matrix, gamma, _, _ = fem.condense(mass_matrix, gamma, D=mesh.boundary_nodes())
    
    return Vh, stiffness_matrix, gamma, mass_matrix


"""
Functions for initializing the Q tensor at time 0
"""
def get_n0(x, y, ic=1):
    if ic == 1:
        #n0 = (2-x)*(x+2)*(2-y)*(y+2) / 8
        #n1 = np.sin(np.pi*x)*np.sin(np.pi*y)

        n0 = (2-x)*(2+x)*(2-y)*(2+y)#(2-x)**2*(x+2)*(2-y)*(y+2)**2 / 50
        n1 = np.sin(np.pi*x)*np.sin(np.pi*y)
    else:
        raise ValueError("Invalid initial condition")
    
    n0 = n0 / np.sqrt(1 + n0**2 + n1**2)
    n1 = n1 / np.sqrt(1 + n0**2 + n1**2)

    return np.array([[n0], [n1]])

def get_Q0(n0):
    return n0 @ n0.T - np.sum(n0*n0)/2.0 * np.eye(2)


def initialize_Q_flow(interior_point_coords, Nt, a, b, c, A0):
    """
    returns: Q1, Q2, p1, p2, r
    """

    num_interior_points = interior_point_coords.shape[0]

    Q1 = np.zeros((Nt, num_interior_points))
    Q2 = np.zeros((Nt, num_interior_points))

    p1 = np.zeros((Nt, num_interior_points))
    p2 = np.zeros((Nt, num_interior_points))

    r  = np.zeros((Nt, num_interior_points))

    index = 0
    for i, point in enumerate(interior_point_coords):

        x, y = point[0], point[1]

        n0_ij = get_n0(x, y)
        Q0_ij = get_Q0(n0_ij)

        Q0sq_ij = Q0_ij @ Q0_ij
        r[0, index] = np.sqrt(2 * ((a/2)*np.trace(Q0sq_ij) \
            - (b/3)*np.trace(Q0sq_ij @ Q0_ij) + (c/4)*np.trace(Q0sq_ij)**2 + A0))
        Q1[0, index] = Q0_ij[0, 0] # store the top left entry of Q0_ij
        Q2[0, index] = Q0_ij[0, 1] # store the top right entry of Q0_ij

        # make the initial P^{n+1/2} value P(Q0)
        P0_ij = P(Q0_ij, a, b, c, A0)
        p1[0, index] = P0_ij[0, 0] # store the top left entry of P0_ij
        p2[0, index] = P0_ij[0, 1] # store the top right entry of P0_ij

        index += 1
    
    return Q1, Q2, p1, p2, r, num_interior_points


"""
P(Q) used in the definition of the scheme
"""
def P(Q, a, b, c, A0):
    """ 
    Q (np.array): Nx2x2 Q-tensor
    returns: P(Q) as defined in the paper
    """
    if len(Q.shape) == 2:
        Q2 = np.matmul(Q, Q)
        trQ2 = np.trace(Q2)
        SQ = a*Q - b*(Q2 - 1/2 * trQ2*np.eye(2)) + c*trQ2*Q
        rQ = np.sqrt(2*(a/2 * trQ2 - b/3*np.trace(np.matmul(Q2, Q)) + c/4*trQ2**2 + A0))

        return SQ / rQ
    else:
        Q2 = np.matmul(Q, Q)
        trQ2 = np.trace(Q2, axis1=1, axis2=2)
        SQ = a*Q - b*(Q2 - 1/2 * trQ2.reshape(-1,1,1)*np.eye(2)) + c*trQ2.reshape(-1,1,1)*Q
        rQ = np.sqrt(2*(a/2 * trQ2 - b/3*np.trace(np.matmul(Q2, Q), axis1=1, axis2=2) + c/4*trQ2**2 + A0)).reshape(-1,1,1)
        return SQ / rQ

def update_P(time, Q1, Q2, p1, p2, a, b, c, A0):
    Q = np.concatenate((
            np.concatenate((Q1[:, None, None], Q2[:, None, None]), axis=2),
            np.concatenate((Q2[:, None, None], -Q1[:, None, None]), axis=2),
    ), axis=1)
    PQ = P(Q, a, b, c, A0)
    p1[time] = PQ[:, 0, 0]
    p2[time] = PQ[:, 0, 1]

    
"""
Functions for experiments (e.g. visualization, energy, etc.)
"""
def visualize(points, Q1, Q2, r, num_interior_points, Nt, every=10):
    director1x = np.zeros((Nt, points.shape[0]))
    director1y = np.zeros((Nt, points.shape[0]))
    director2x = np.zeros((Nt, points.shape[0]))
    director2y = np.zeros((Nt, points.shape[0]))
    
    Q = np.zeros((Nt, num_interior_points, 2, 2))

    for t in tqdm(range(Nt)):
        for i in range(points.shape[0]):
            
            Q[t, i, 0, 0] = Q1[t, i]
            Q[t, i, 0, 1] = Q2[t, i]
            Q[t, i, 1, 0] = Q2[t, i]
            Q[t, i, 1, 1] = -Q1[t, i]

            eigenvalues, eigenvectors = np.linalg.eigh(Q[t, i])

            v1 = eigenvectors[:, 0]
            v2 = eigenvectors[:, 1]
            director1x[t, i] = v1[0]
            director1y[t, i] = v1[1]
            director2x[t, i] = v2[0]
            director2y[t, i] = v2[1]
        
        if t % every == 0:
            plt.clf()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
            plt.quiver(points[:, 0], points[:, 1], director2x[t], director2y[t], headaxislength=0, headwidth=0, headlength=0, color='b', pivot='mid')
            plt.pause(0.4)
            #plt.show()
    
    plt.show()

    
def get_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix):
    energy = []
    expected_energy = []
    
    L1 = L

    tVals = np.linspace(0, t_final, Nt)
    dt = tVals[1] - tVals[0]

    
    # iterate over n + 1
    for np1 in range(Nt):
        
        #### first calculate norm of D_t^+ Q^n ####
        if np1 == 0:
            # we assume that the time -1 is same as time 0
            norm_D_t_plus_Qsq = 0
        else:
            D_t_plus_Q1 = (Q1[np1] - Q1[np1-1]) / dt
            D_t_plus_Q2 = (Q2[np1] - Q2[np1-1]) / dt
            
            norm_D_t_plus_Qsq = np.sum(2 * (D_t_plus_Q1**2 + D_t_plus_Q2**2) * gamma)

        #### calculate norm of grad Q^{n+1} ####
        norm_grad_Qsq = np.sum(2 * (
            Q1[np1].reshape(1, -1) @ stiffness_matrix @ Q1[np1].reshape(-1, 1) + \
            Q2[np1].reshape(1, -1) @ stiffness_matrix @ Q2[np1].reshape(-1, 1)
        ))
        #print(norm_grad_Qsq)

        #### calculate norm of r^{n+1} ####
        norm_rsq = np.sum(r[np1]**2 * gamma)
        
        assert norm_D_t_plus_Qsq >= 0 and norm_grad_Qsq >= 0 and norm_rsq >= 0

        energy.append(
            sigma/2 * norm_D_t_plus_Qsq + M*L1/2 * norm_grad_Qsq + M/2 * norm_rsq
        )

        if np1 <= 1:
            expected_energy.append(energy[-1])
        else:
            D_t_nm1_Q1 = (Q1[np1-1] - Q1[np1-2]) / dt
            D_t_nm1_Q2 = (Q2[np1-1] - Q2[np1-2]) / dt

            norm_D_t_diff_sq = np.sum(2 * ((D_t_plus_Q1 - D_t_nm1_Q1)**2 + (D_t_plus_Q2 - D_t_nm1_Q2)**2) * gamma)

            expected_energy.append(
                expected_energy[-1] - dt * norm_D_t_plus_Qsq - sigma/2 * norm_D_t_diff_sq
            )

    return energy, expected_energy

def plot_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix):
    energy, expected = get_energy(Q1, Q2, r, Lx, Nt, sigma, M, L, t_final, A0, gamma, stiffness_matrix)
    plt.plot(np.arange(len(energy)), energy, label="Energy")
    plt.plot(np.arange(len(energy)), expected, label="Expected Energy")
    plt.legend()
    plt.show()
    
    plt.plot(np.arange(len(energy)), np.abs(np.array(energy) - np.array(expected)))
    plt.title("|Expected Energy - Energy|")
    plt.show()

    return energy