import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as la
from scipy import interpolate
from tqdm import tqdm
import time

from other_functions import *

def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 30)
        #self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, out_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        #x = self.fc2(x)
        #x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

class Net2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(in_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, out_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

def NEIM_Q(ro_sols, f_NEIM, mu, max_modes=10, train_loop_iterations=10000, theta_train_loop_iterations=20000):
    NUM_PARAMS = f_NEIM.shape[0]
    RO_DIM = ro_sols.shape[1] // 3
    
    #WEIGHTS = np.array([[(abs(i - j) <= 10)/(1+abs(i-j)**2) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[(abs(i - j) <= 10)/(1+10000*abs(i-j)**2) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[np.exp(-100*np.abs(mu[i] - mu[j])) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[(i == j)*1.0 for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.ones((mu.shape[0], mu.shape[0]))
    WEIGHTS = np.zeros((mu.shape[0], mu.shape[0]))
    max_time = np.max(mu[:, 0])
    for i in range(mu.shape[0]):
        a_i = mu[i, 1]
        for j in range(mu.shape[0]):
            a_j = mu[j, 1]
            if abs(a_i - a_j) < 1e-6 and abs(i-j) <= 5:
                WEIGHTS[i, j] = 1
    #WEIGHTS = np.ones((mu.shape[0], mu.shape[0]))
    
    selected_indices = []
    trained_networks = []
    normalizations = []
    
    # NEIM Step 1
    errors = np.array([np.sum(f_NEIM[i, i]**2) for i in range(NUM_PARAMS)])
    idx = np.argmax(errors)
    selected_indices.append(idx)
    print(idx, "Max Error:", errors[idx], "Mean Error:", np.mean(errors))
    
    Network_mu_1 = Net(3*RO_DIM, 2*RO_DIM)
    mse_weights = torch.tensor(WEIGHTS[idx].reshape(-1, 1))
    optimizer = torch.optim.Adam(Network_mu_1.parameters(), lr=0.001)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    
    x_mean = np.mean(ro_sols, axis=0, keepdims=True)
    x_std  = np.std(ro_sols, axis=0, keepdims=True)
    x_std[x_std == 0] = 1
    x_data = torch.tensor((ro_sols - x_mean) / x_std).float()
    
    y_data = np.array([f_NEIM[idx, j]/np.linalg.norm(f_NEIM[idx, j]) for j in range(f_NEIM.shape[1])])
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    y_std[y_std == 0] = 1
    y_data = torch.tensor((y_data - y_mean) / y_std).float()
    normalizations.append((y_mean, y_std))
    
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
        numerator = 0
        denominator = 0
        for j in range(NUM_PARAMS):
            if i == j:
                net_u_mu = (Network_mu_1(x_data[j].view(1, -1)).detach().numpy().reshape(1, -1) * y_std + y_mean).reshape(-1)
                numerator += WEIGHTS[i, j] * np.dot(f_NEIM[i, j], net_u_mu)
                denominator += WEIGHTS[i, j] * np.dot(net_u_mu, net_u_mu)

        theta_1_1_i = numerator / denominator
        thetas[i] = theta_1_1_i    
    
    for iteration in range(max_modes-1):
        # current approximation
        approx = lambda sol, param_idx: sum([
            thetas[param_idx][i] * (net(torch.tensor((sol.reshape(1,-1) - x_mean) / x_std).float()).T.detach().numpy().reshape(1,-1) * normalizations[i][1] + normalizations[i][0]).reshape(-1)\
            for i, net in enumerate(trained_networks)
        ])
    
        
        # Compute Errors and Choose New Parameter
        errors = np.array([np.sum((f_NEIM[i, i] - approx(ro_sols[i], i))**2) for i in range(NUM_PARAMS)])
        print("Mean Already Selected Error:", np.mean(errors[np.array(selected_indices)]))
        errors[np.array(selected_indices)] = -1
        mu_2_idx = np.argmax(errors)
        selected_indices.append(mu_2_idx)
        print(mu_2_idx, "Max Error:", errors[mu_2_idx], "Mean Error:", np.mean(errors[errors>=0]))
        
        Network_mu_2 = Net(3*RO_DIM, 2*RO_DIM)
        mse_weights = torch.tensor(WEIGHTS[mu_2_idx].reshape(-1, 1))
        optimizer = torch.optim.Adam(Network_mu_2.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)


        # need to do Gram-Schmidt on this matrix
        previous_y_data = np.copy(y_data)
        y_data = np.copy(f_NEIM[mu_2_idx])
        
        for net_idx, net in enumerate(trained_networks):
            # form matrix of evaluations for this network
            previous_net_matrix = np.zeros((NUM_PARAMS, f_NEIM.shape[2]))
            for i in range(NUM_PARAMS):
                previous_net_matrix[i] = (net(x_data[i].view(1, -1)).detach().numpy().reshape(1,-1) * normalizations[net_idx][1] + normalizations[net_idx][0]).reshape(-1)
                y_data[i] -= np.dot(y_data[i], previous_net_matrix[i]) * previous_net_matrix[i] / np.linalg.norm(previous_net_matrix[i])**2
        
        for j in range(f_NEIM.shape[1]):
             y_data[j] = y_data[j] / np.linalg.norm(y_data[j])
        
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        y_std[y_std == 0] = 1
        y_data = torch.tensor((y_data - y_mean) / y_std).float()
        normalizations.append((y_mean, y_std))
        
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
            LHS = np.zeros((num_nets, num_nets))
            RHS = np.zeros((num_nets, 1))

            for j in range(NUM_PARAMS):
                if i == j:
                    nets_u_mu = [(net(x_data[j].view(1, -1)).detach().numpy().reshape(1,-1) * normalizations[idx][1] + normalizations[idx][0]).reshape(-1) for idx, net in enumerate(trained_networks)]
                    for k1 in range(num_nets):
                        RHS[k1] += WEIGHTS[i, j] * np.dot(f_NEIM[i, j], nets_u_mu[k1])
                        for k2 in range(num_nets):
                            LHS[k1, k2] += WEIGHTS[i, j] * np.dot(nets_u_mu[k1], nets_u_mu[k2])

            thetas[i] = np.linalg.solve(LHS, RHS).reshape(-1)
        
    # train neural network to predict thetas
    theta_net = Net2(mu.shape[1], num_nets)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(theta_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)

    mu_mean = np.mean(mu, axis=0, keepdims=True)
    mu_std = np.std(mu, axis=0, keepdims=True)
    mu_std[mu_std == 0] = 1
    
    x_data = torch.tensor((mu - mu_mean) / mu_std).float()
    y_data = torch.tensor(thetas).float()

    for epoch in range(theta_train_loop_iterations):
        optimizer.zero_grad()
        output = theta_net(x_data)
        loss = criterion(output, y_data)
        if epoch % 100 == 0:
            print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    theta_net.eval()
        
        
    def NEIM_approximation(new_mu, new_sol, num_modes=-1):
        if num_modes == -1:
            num_modes = len(trained_networks)
        
        new_mu = torch.tensor((new_mu.reshape(1,-1) - mu_mean) / mu_std).float()
        new_sol = torch.tensor((new_sol.reshape(1,-1) - x_mean) / x_std)
        thetas_ = theta_net(new_mu).view(-1).detach().numpy()
        s = 0
        for i, net in enumerate(trained_networks[:num_modes]):
            s += thetas_[i] * (net(new_sol.view(1, -1).float()).view(1,-1).detach().numpy() * normalizations[i][1] + normalizations[i][0]).reshape(-1)
        
        return s.reshape(-1)
    
    return NEIM_approximation, selected_indices, trained_networks, mu, thetas

def NEIM_R(ro_sols, f_NEIM, mu, max_modes=10, train_loop_iterations=10000, theta_train_loop_iterations=20000):
    NUM_PARAMS = f_NEIM.shape[0]
    RO_DIM = ro_sols.shape[1] // 4
    
    #WEIGHTS = np.array([[(abs(i - j) <= 10)/(1+abs(i-j)**2) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[(abs(i - j) <= 10)/(1+10000*abs(i-j)**2) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[np.exp(-100*np.abs(mu[i] - mu[j])) for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    #WEIGHTS = np.array([[(i == j)*1.0 for j in range(mu.shape[0])] for i in range(mu.shape[0])])
    WEIGHTS = np.zeros((mu.shape[0], mu.shape[0]))
    max_time = np.max(mu[:, 0])
    for i in range(mu.shape[0]):
        a_i = mu[i, 1]
        for j in range(mu.shape[0]):
            a_j = mu[j, 1]
            if abs(a_i - a_j) < 1e-6 and abs(i-j) <= 5: # a_i - a_j was <= 0.1
                WEIGHTS[i, j] = 1
    #WEIGHTS = np.ones((mu.shape[0], mu.shape[0]))
    
    selected_indices = []
    trained_networks = []
    normalizations = []
    
    # NEIM Step 1
    errors = np.array([np.sum(f_NEIM[i, i]**2) for i in range(NUM_PARAMS)])
    idx = np.argmax(errors)
    selected_indices.append(idx)
    print(idx, "Max Error:", errors[idx], "Mean Error:", np.mean(errors))
    
    Network_mu_1 = Net(4*RO_DIM, RO_DIM)
    mse_weights = torch.tensor(WEIGHTS[idx].reshape(-1, 1))
    optimizer = torch.optim.Adam(Network_mu_1.parameters(), lr=0.001)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    
    x_mean = np.mean(ro_sols, axis=0, keepdims=True)
    x_std  = np.std(ro_sols, axis=0, keepdims=True)
    x_std[x_std == 0] = 1
    x_data = torch.tensor((ro_sols - x_mean) / x_std).float()
    
    y_data = np.array([f_NEIM[idx, j]/np.linalg.norm(f_NEIM[idx, j]) for j in range(f_NEIM.shape[1])])
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    y_std[y_std == 0] = 1
    y_data = torch.tensor((y_data - y_mean) / y_std).float()
    normalizations.append((y_mean, y_std))
    
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
        numerator = 0
        denominator = 0
        for j in range(NUM_PARAMS):
            if i == j:
                net_u_mu = (Network_mu_1(x_data[j].view(1, -1)).detach().numpy().reshape(1, -1) * y_std + y_mean).reshape(-1)
                numerator += WEIGHTS[i, j] * np.dot(f_NEIM[i, j], net_u_mu)
                denominator += WEIGHTS[i, j] * np.dot(net_u_mu, net_u_mu)

        theta_1_1_i = numerator / denominator
        thetas[i] = theta_1_1_i    
    
    for iteration in range(max_modes-1):
        # current approximation
        approx = lambda sol, param_idx: sum([
            thetas[param_idx][i] * (net(torch.tensor((sol.reshape(1,-1) - x_mean) / x_std).float()).T.detach().numpy().reshape(1,-1) * normalizations[i][1] + normalizations[i][0]).reshape(-1)\
            for i, net in enumerate(trained_networks)
        ])
    
        
        # Compute Errors and Choose New Parameter
        errors = np.array([np.sum((f_NEIM[i, i] - approx(ro_sols[i], i))**2) for i in range(NUM_PARAMS)])
        print("Mean Already Selected Error:", np.mean(errors[np.array(selected_indices)]))
        errors[np.array(selected_indices)] = -1
        mu_2_idx = np.argmax(errors)
        selected_indices.append(mu_2_idx)
        print(mu_2_idx, "Max Error:", errors[mu_2_idx], "Mean Error:", np.mean(errors[errors>=0]))
        
        Network_mu_2 = Net(4*RO_DIM, RO_DIM)
        mse_weights = torch.tensor(WEIGHTS[mu_2_idx].reshape(-1, 1))
        optimizer = torch.optim.Adam(Network_mu_2.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)


        # need to do Gram-Schmidt on this matrix
        previous_y_data = np.copy(y_data)
        y_data = np.copy(f_NEIM[mu_2_idx])
        
        for net_idx, net in enumerate(trained_networks):
            # form matrix of evaluations for this network
            previous_net_matrix = np.zeros((NUM_PARAMS, f_NEIM.shape[2]))
            for i in range(NUM_PARAMS):
                previous_net_matrix[i] = (net(x_data[i].view(1, -1)).detach().numpy().reshape(1,-1) * normalizations[net_idx][1] + normalizations[net_idx][0]).reshape(-1)
                y_data[i] -= np.dot(y_data[i], previous_net_matrix[i]) * previous_net_matrix[i] / np.linalg.norm(previous_net_matrix[i])**2
        
        for j in range(f_NEIM.shape[1]):
             y_data[j] = y_data[j] / np.linalg.norm(y_data[j])
        
        y_mean = np.mean(y_data, axis=0)
        y_std = np.std(y_data, axis=0)
        y_std[y_std == 0] = 1
        y_data = torch.tensor((y_data - y_mean) / y_std).float()
        normalizations.append((y_mean, y_std))
        
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
            LHS = np.zeros((num_nets, num_nets))
            RHS = np.zeros((num_nets, 1))

            for j in range(NUM_PARAMS):
                if i == j:
                    nets_u_mu = [(net(x_data[j].view(1, -1)).detach().numpy().reshape(1,-1) * normalizations[idx][1] + normalizations[idx][0]).reshape(-1) for idx, net in enumerate(trained_networks)]
                    for k1 in range(num_nets):
                        RHS[k1] += WEIGHTS[i, j] * np.dot(f_NEIM[i, j], nets_u_mu[k1])
                        for k2 in range(num_nets):
                            LHS[k1, k2] += WEIGHTS[i, j] * np.dot(nets_u_mu[k1], nets_u_mu[k2])

            thetas[i] = np.linalg.solve(LHS, RHS).reshape(-1)
        
    # train neural network to predict thetas
    theta_net = Net2(mu.shape[1], num_nets)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(theta_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    
    mu_mean = np.mean(mu, axis=0, keepdims=True)
    mu_std = np.std(mu, axis=0, keepdims=True)
    mu_std[mu_std == 0] = 1
    
    x_data = torch.tensor((mu - mu_mean) / mu_std).float()
    y_data = torch.tensor(thetas).float()

    for epoch in range(theta_train_loop_iterations):
        optimizer.zero_grad()
        output = theta_net(x_data)
        loss = criterion(output, y_data)
        if epoch % 100 == 0:
            print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    theta_net.eval()
        
        
    def NEIM_approximation(new_mu, new_sol, num_modes=-1):
        if num_modes == -1:
            num_modes = len(trained_networks)
        
        new_mu = torch.tensor((new_mu.reshape(1,-1) - mu_mean) / mu_std).float()
        new_sol = torch.tensor((new_sol.reshape(1,-1) - x_mean) / x_std)
        thetas_ = theta_net(new_mu).view(-1).detach().numpy()
        s = 0
        for i, net in enumerate(trained_networks[:num_modes]):
            s += thetas_[i] * (net(new_sol.view(1, -1).float()).view(1,-1).detach().numpy() * normalizations[i][1] + normalizations[i][0]).reshape(-1)
        
        return s.reshape(-1)
    
    return NEIM_approximation, selected_indices, trained_networks, mu, thetas

def initialize_Q_flow_NEIM(Q1, Q2, r, U_Q1, U_Q2, U_r):
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
    
    r_ = np.zeros((U_r.shape[1], r.shape[0]))
    r_[:, 0] = (U_r.T @ r[0].reshape(-1, 1)).reshape(-1)
    r_ = r_.T
    
    return Q1_, Q2_, r_

def solve_Q_flow_linear_system_NEIM(Q1, Q2, r, LHS, RHS, NQ_NEIM, time, a, b, c):
    """
    returns: Q1[time], Q2[time]
    """
    ro_sol = np.concatenate([Q1, Q2, r])
    nonlinearityQ = NQ_NEIM(np.array([time, a]), ro_sol).reshape(-1, 1)#NQ_NEIM(np.array([time, a, b, c]), ro_sol).reshape(-1, 1)
    g = RHS @ np.concatenate((Q1.reshape(-1, 1), Q2.reshape(-1, 1)), axis=0) - nonlinearityQ
    Q_ = np.linalg.solve(LHS, g).reshape(-1)

    return Q_

def solve_Q_flow_NEIM(Q1, Q2, r, 
                      gamma, stiffness_matrix, 
                      U_Q1, U_Q2, NQ_NEIM, NR_NEIM,
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
        Q_ = solve_Q_flow_linear_system_NEIM(Q1[time-1], Q2[time-1], r[time-1], LHS, RHS, NQ_NEIM,
                                             time, a, b, c)
        Q1[time] = Q_[:Q1_dim]
        Q2[time] = Q_[Q1_dim:]

        # update r
        ro_sol = np.concatenate([Q1[time], Q1[time-1], Q2[time], Q2[time-1]])
        r[time] = r[time-1] + NR_NEIM(np.array([time-1, a]), ro_sol)#NR_NEIM(np.array([time-1, a, b, c]), ro_sol)