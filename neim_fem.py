import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as la
from scipy import interpolate
from tqdm import tqdm

from other_functions import *

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, out_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

def NEIM_Q(ro_sols, f_NEIM, mu, max_modes=10, train_loop_iterations=10000):
    """
    ro_sols: (number of parameters, number of times, 3 * ro dim)
    f_NEIM: (number of parameters, number of times, 2 * ro dim)
    """
    NUM_PARAMS = ro_sols.shape[0]
    NUM_TIMES = ro_sols.shape[1]
    RO_DIM = ro_sols.shape[2] // 3
    
    WEIGHTS = np.ones((f_NEIM.shape[0], f_NEIM.shape[1]))
    #WEIGHTS[:, 0] = 1
    
    selected_indices = []
    trained_networks = []
    
    # NEIM Step 1
    errors = np.array([np.sum(f_NEIM[i]**2) / NUM_TIMES for i in range(NUM_PARAMS)])
    idx = np.argmax(errors)
    selected_indices.append(idx)
    print(idx, "Max Error:", errors[idx], "Mean Error:", np.mean(errors))
    
    Network_mu_1 = Net(3 * RO_DIM, 2 * RO_DIM)
    mse_weights = torch.tensor(WEIGHTS[idx].reshape(-1, 1))
    optimizer = torch.optim.Adam(Network_mu_1.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    
    x_data = torch.tensor(ro_sols[idx]).float()
    y_data = torch.tensor(np.array([f_NEIM[idx, time] for time in range(f_NEIM.shape[1])])).float()
    
    for epoch in range(train_loop_iterations):
        optimizer.zero_grad()
        output = Network_mu_1(x_data)
        loss = weighted_mse_loss(output, y_data, mse_weights)
        if epoch % 100 == 0:
            print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    
    Network_mu_1.eval()
    trained_networks.append(Network_mu_1)
    
    thetas = np.zeros((NUM_PARAMS, 1))
    for i in range(NUM_PARAMS):
        x_data = torch.tensor(ro_sols[i]).float()
        numerator = 0
        denominator = 0
        for time in range(NUM_TIMES):
            net_u_mu = Network_mu_1(x_data[time].view(1, -1)).detach().numpy().reshape(-1)
            numerator += WEIGHTS[i, time] * np.dot(f_NEIM[i, time], net_u_mu)
            denominator += WEIGHTS[i, time] * np.dot(net_u_mu, net_u_mu)

        theta_1_1_i = numerator / denominator
        thetas[i] = theta_1_1_i
    
    for iteration in range(max_modes-1):
        # current approximation
        approx = lambda sol, param_idx: sum([
            thetas[param_idx][i] * net(torch.tensor(sol).view(1, -1).float()).T.detach().numpy().reshape(-1)\
            for i, net in enumerate(trained_networks)
        ])
        
        
        # Compute Errors and Choose New Parameter
        errors = []
        for i in range(NUM_PARAMS):
            errors.append(0)
            for time in range(NUM_TIMES):
                errors[-1] += np.sum((f_NEIM[i, time] - approx(ro_sols[i, time], i))**2)
            errors[-1] /= NUM_TIMES
        errors = np.array(errors)
        print("Mean Already Selected Error:", np.mean(errors[np.array(selected_indices)]))
        errors[np.array(selected_indices)] = -1
        mu_2_idx = np.argmax(errors)
        selected_indices.append(mu_2_idx)
        print(mu_2_idx, "Max Error:", errors[mu_2_idx], "Mean Error:", np.mean(errors[errors>=0]))
        
        Network_mu_2 = Net(3 * RO_DIM, 2 * RO_DIM)
        mse_weights = torch.tensor(WEIGHTS[mu_2_idx].reshape(-1, 1))
        optimizer = torch.optim.Adam(Network_mu_2.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)


        # need to do Gram-Schmidt on this matrix
        x_data = torch.tensor(ro_sols[mu_2_idx]).float()
        #previous_y_data = np.copy(y_data)
        y_data = np.copy(f_NEIM[mu_2_idx])
        
        for net in trained_networks:
            # form matrix of evaluations for this network
            previous_net_matrix = np.zeros((NUM_TIMES, f_NEIM.shape[2]))
            for time in range(NUM_TIMES):
                previous_net_matrix[time] = net(x_data[time].view(1, -1)).detach().numpy().reshape(-1)
                y_data[time] -= np.dot(y_data[time], previous_net_matrix[time]) * previous_net_matrix[time] / np.linalg.norm(previous_net_matrix[time])**2#previous_net_matrix[time]#
        
        #for time in range(f_NEIM.shape[1]):
        #    y_data[time] = y_data[time] / np.linalg.norm(y_data[time])
        
        y_data = torch.tensor(y_data).float()
        
        for epoch in range(train_loop_iterations):
            optimizer.zero_grad()
            output = Network_mu_2(x_data)
            loss = weighted_mse_loss(output, y_data, mse_weights)
            if epoch % 100 == 0:
                print(epoch, loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        Network_mu_2.eval()
        trained_networks.append(Network_mu_2)
        
        # 3c. Find theta_1_2(mu), theta_2_2(mu)
        print("\nFinding theta...")
        num_nets = len(trained_networks)
        thetas = np.zeros((NUM_PARAMS, num_nets))
        for i in range(NUM_PARAMS):
            x_data = torch.tensor(ro_sols[i]).float()
            LHS = np.zeros((num_nets, num_nets))
            RHS = np.zeros((num_nets, 1))

            for time in range(NUM_TIMES):
                nets_u_mu = [net(x_data[time].view(1, -1)).detach().numpy().reshape(-1) for net in trained_networks]
                for k1 in range(num_nets):
                    RHS[k1] += WEIGHTS[i, time] * np.dot(f_NEIM[i, time], nets_u_mu[k1])
                    for k2 in range(num_nets):
                        LHS[k1, k2] += WEIGHTS[i, time] * np.dot(nets_u_mu[k1], nets_u_mu[k2])

            thetas[i] = np.linalg.solve(LHS, RHS).reshape(-1)
            print(thetas[i])
        
    def NEIM_approximation(new_mu, new_sol, num_modes=-1):
        if num_modes == -1:
            num_modes = len(trained_networks)
        
        new_sol = torch.tensor(new_sol)
        thetas_ = interpolate.griddata(mu, thetas, new_mu, method='linear').reshape(-1)
        if True in np.isnan(thetas_):
            thetas_ = interpolate.griddata(mu, thetas, new_mu, method='nearest').reshape(-1)
        s = 0
        for i, net in enumerate(trained_networks[:num_modes]):
            s += thetas_[i] * net(new_sol.view(1, -1).float()).view(-1).detach().numpy()
        return s.reshape(-1)
    
    return NEIM_approximation, selected_indices, trained_networks, mu, thetas

def NEIM_R(ro_sols, f_NEIM, mu, max_modes=10, train_loop_iterations=10000):
    """
    ro_sols: (number of parameters, number of times, 4 * ro dim)
    f_NEIM: (number of parameters, number of times, ro dim)
    """
    NUM_PARAMS = ro_sols.shape[0]
    NUM_TIMES = ro_sols.shape[1]
    RO_DIM = ro_sols.shape[2] // 4
    
    WEIGHTS = np.ones((f_NEIM.shape[0], f_NEIM.shape[1]))
    #WEIGHTS[:, 0] = 1
    
    selected_indices = []
    trained_networks = []
    
    # NEIM Step 1
    errors = np.array([np.sum(f_NEIM[i]**2) / NUM_TIMES for i in range(NUM_PARAMS)])
    idx = np.argmax(errors)
    selected_indices.append(idx)
    print(idx, "Max Error:", errors[idx], "Mean Error:", np.mean(errors))
    
    Network_mu_1 = Net(4 * RO_DIM, RO_DIM)
    mse_weights = torch.tensor(WEIGHTS[idx].reshape(-1, 1))
    optimizer = torch.optim.Adam(Network_mu_1.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    
    x_data = torch.tensor(ro_sols[idx]).float()
    y_data = torch.tensor(np.array([f_NEIM[idx, time] for time in range(f_NEIM.shape[1])])).float()
    
    for epoch in range(train_loop_iterations):
        optimizer.zero_grad()
        output = Network_mu_1(x_data)
        loss = weighted_mse_loss(output, y_data, mse_weights)
        if epoch % 100 == 0:
            print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    
    Network_mu_1.eval()
    trained_networks.append(Network_mu_1)
    
    thetas = np.zeros((NUM_PARAMS, 1))
    for i in range(NUM_PARAMS):
        x_data = torch.tensor(ro_sols[i]).float()
        numerator = 0
        denominator = 0
        for time in range(NUM_TIMES):
            net_u_mu = Network_mu_1(x_data[time].view(1, -1)).detach().numpy().reshape(-1)
            numerator += WEIGHTS[i, time] * np.dot(f_NEIM[i, time], net_u_mu)
            denominator += WEIGHTS[i, time] * np.dot(net_u_mu, net_u_mu)

        theta_1_1_i = numerator / denominator
        thetas[i] = theta_1_1_i
    
    for iteration in range(max_modes-1):
        # current approximation
        approx = lambda sol, param_idx: sum([
            thetas[param_idx][i] * net(torch.tensor(sol).view(1, -1).float()).T.detach().numpy().reshape(-1)\
            for i, net in enumerate(trained_networks)
        ])
        
        
        # Compute Errors and Choose New Parameter
        errors = []
        for i in range(NUM_PARAMS):
            errors.append(0)
            for time in range(NUM_TIMES):
                #print(i, time, np.linalg.norm(f_NEIM[i, time]), np.linalg.norm(approx(ro_sols[i, time], i)))
                errors[-1] += np.sum((f_NEIM[i, time] - approx(ro_sols[i, time], i))**2)
            errors[-1] /= NUM_TIMES
        errors = np.array(errors)
        print("Mean Already Selected Error:", np.mean(errors[np.array(selected_indices)]))
        errors[np.array(selected_indices)] = -1
        mu_2_idx = np.argmax(errors)
        selected_indices.append(mu_2_idx)
        print(mu_2_idx, "Max Error:", errors[mu_2_idx], "Mean Error:", np.mean(errors[errors>=0]))
        
        Network_mu_2 = Net(4 * RO_DIM, RO_DIM)
        mse_weights = torch.tensor(WEIGHTS[mu_2_idx].reshape(-1, 1))
        optimizer = torch.optim.Adam(Network_mu_2.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)


        # need to do Gram-Schmidt on this matrix
        x_data = torch.tensor(ro_sols[mu_2_idx]).float()
        #previous_y_data = np.copy(y_data)
        y_data = np.copy(f_NEIM[mu_2_idx])
        
        for net in trained_networks:
            # form matrix of evaluations for this network
            previous_net_matrix = np.zeros((NUM_TIMES, f_NEIM.shape[2]))
            for time in range(NUM_TIMES):
                previous_net_matrix[time] = net(x_data[time].view(1, -1)).detach().numpy().reshape(-1)
                #y_data[time] -= np.dot(y_data[time], previous_net_matrix[time]) * previous_net_matrix[time] / np.linalg.norm(previous_net_matrix[time])**2#previous_net_matrix[time]#
        
        #for time in range(f_NEIM.shape[1]):
        #    y_data[time] = y_data[time] / np.linalg.norm(y_data[time])
        
        y_data = torch.tensor(y_data).float()
        
        for epoch in range(train_loop_iterations):
            optimizer.zero_grad()
            output = Network_mu_2(x_data)
            #print(output, y_data)
            loss = weighted_mse_loss(output, y_data, mse_weights)
            if epoch % 100 == 0:
                print(epoch, loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        Network_mu_2.eval()
        trained_networks.append(Network_mu_2)
        
        # 3c. Find theta_1_2(mu), theta_2_2(mu)
        print("\nFinding theta...")
        num_nets = len(trained_networks)
        thetas = np.zeros((NUM_PARAMS, num_nets))
        for i in range(NUM_PARAMS):
            x_data = torch.tensor(ro_sols[i]).float()
            LHS = np.zeros((num_nets, num_nets))
            RHS = np.zeros((num_nets, 1))

            for time in range(NUM_TIMES):
                nets_u_mu = [net(x_data[time].view(1, -1)).detach().numpy().reshape(-1) for net in trained_networks]
                for k1 in range(num_nets):
                    RHS[k1] += WEIGHTS[i, time] * np.dot(f_NEIM[i, time], nets_u_mu[k1])
                    for k2 in range(num_nets):
                        LHS[k1, k2] += WEIGHTS[i, time] * np.dot(nets_u_mu[k1], nets_u_mu[k2])

            thetas[i] = np.linalg.solve(LHS, RHS).reshape(-1)
            print(thetas[i])
        
    def NEIM_approximation(new_mu, new_sol, num_modes=-1):
        if num_modes == -1:
            num_modes = len(trained_networks)
        
        new_sol = torch.tensor(new_sol)
        thetas_ = interpolate.griddata(mu, thetas, new_mu, method='linear').reshape(-1)
        if True in np.isnan(thetas_):
            thetas_ = interpolate.griddata(mu, thetas, new_mu, method='nearest').reshape(-1)
        s = 0
        for i, net in enumerate(trained_networks[:num_modes]):
            s += thetas_[i] * net(new_sol.view(1, -1).float()).view(-1).detach().numpy()
        return s.reshape(-1)
    
    return NEIM_approximation, selected_indices, trained_networks, mu, thetas

def initialize_Q_flow_NEIM(Q1, Q2, p1, p2, r, U_Q1, U_Q2, U_r, nQ_indices, nR_indices):
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

def solve_Q_flow_linear_system_NEIM(Q1, Q2, p1_Q, p2_Q, r, gamma, U_r, U_deimQ1, U_deimQ2, nQ_indices, LHS, RHS, M):
    """
    returns: Q1[time], Q2[time]
    """

    g1 = M * U_deimQ1 @ (gamma[nQ_indices] * p1_Q * (U_r[nQ_indices] @ r))
    g2 = M * U_deimQ2 @ (gamma[nQ_indices] * p2_Q * (U_r[nQ_indices] @ r))
    g = RHS @ np.concatenate((Q1.reshape(-1, 1), Q2.reshape(-1, 1)), axis=0) - np.concatenate((g1, g2), axis=0)

    Q_ = np.linalg.solve(LHS, g).reshape(-1)

    return Q_

def solve_Q_flow_NEIM(Q1, Q2, p1_Q, p2_Q, p1_R, p2_R, r, 
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