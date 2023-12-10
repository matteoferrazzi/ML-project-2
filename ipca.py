import numpy as np
import pandas as pd

def gamma_first(ret, data, k):    
    x = []
    for t in range(len(ret)):
        x.append(data[t].values.T@ret[t].values/len(ret[0]))

    x_cov = np.sum([x[i].reshape((-1,1))@x[i].reshape((1,-1)) for i in range(len(x))], axis = 0)
    eigval_x, eigvec_x = np.linalg.eig(x_cov)

    idx = np.argsort(np.abs(eigval_x))[::-1]
    sort_eigvec_x = eigvec_x[:,idx].real
    gamma_first = sort_eigvec_x[:,:k]

    return gamma_first

def solve_f(ret, data, gamma, idx):
    # risolve per la singola f poi dobbiamo metterle in una lista
    return np.linalg.solve(gamma.T@data[idx].values.T@data[idx].values@gamma, gamma.T@data[idx].values.T@ret[idx].values)

def solve_gamma(ret, data, f):
    # f viene passato come lista
    A = np.sum([np.kron(data[i].values.T@data[i].values, f[i].reshape(-1,1)@f[i].reshape(1,-1)) for i in range(len(data))], axis=0)
    B = np.sum([np.kron(data[i].values,f[i].reshape((1,-1))).T@ret[i] for i in range(len(data))], axis=0)
    vec_gamma = np.linalg.solve(A, B)
    return vec_gamma.reshape((94, len(f[0])))

def ipca(data, ret, gamma, max_iter):
    
    first = False 
    j = 0
    while j < max_iter:

        j+=1
        temp = []
        f_list_new = []

        for i in range(len(data)):
            f = solve_f(ret, data, gamma, i)
            f_list_new.append(f)
            if first:
                f_change = f-f_list[i]
                temp.append(np.max(f_change))
        first = True
        f_list = f_list_new.copy()

        gamma_new = solve_gamma(ret, data, f_list)
        gamma_change = np.abs(gamma_new-gamma)
        temp.append(np.max(gamma_change))
        gamma = gamma_new.copy()
        
        if (max(temp)<=1e-3):
            break

    return gamma, f_list

def solve_f_reg_w(ret, data, gamma, idx, lambda_, W ):
    # risolve per la singola f poi dobbiamo metterle in una lista
    return np.linalg.solve(gamma.T@data[idx].values.T@W@data[idx].values@gamma + lambda_*np.eye(gamma.shape[1]), 
                           gamma.T@data[idx].values.T@W@ret[idx].values)

def solve_gamma_reg_w(ret, data, f, lambda_, W, gamma):
    # f viene passato come lista
    A = np.sum([np.kron(data[i].values.T@W[i]@data[i].values, f[i].reshape(-1,1)@f[i].reshape(1,-1)) for i in range(len(data))], 
               axis=0) + lambda_*np.eye(gamma.shape[0]*gamma.shape[1])
    
    B = np.sum([np.kron(np.sqrt(W[i])@data[i].values,f[i].reshape((1,-1))).T@np.sqrt(W[i])@ret[i] for i in range(len(data))], axis=0)

    vec_gamma = np.linalg.solve(A, B)
    return vec_gamma.reshape((94, len(f[0])))

def ipca_reg_w(data, ret, gamma_reg_w, max_iter, lambda1, lambda2, W_list):

    first = False 
    j = 0
    while j < max_iter:

        j+=1
        temp = []
        f_list_new = []

        for i in range(len(data)):
            f = solve_f_reg_w(ret, data, gamma_reg_w, i, lambda1, W_list[i])
            f_list_new.append(f)
            if first:
                f_change = f-f_list_reg_w[i]
                temp.append(np.max(f_change))
        first = True
        f_list_reg_w = f_list_new.copy()

        gamma_new = solve_gamma_reg_w(ret, data, f_list_reg_w, lambda2, W_list, gamma_reg_w)
        gamma_change = np.abs(gamma_new-gamma_reg_w)
        temp.append(np.max(gamma_change))
        gamma_reg_w = gamma_new.copy()
        if (max(temp)<=1e-3):
            break

    return gamma_reg_w, f_list_reg_w