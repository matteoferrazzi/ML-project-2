import numpy as np
import pandas as pd

def total_R_squared(ret, data, gamma, f_list):
    sum = 0
    ret_2 = 0
    l = len(ret[0])
    for t in range(len(data)):
        for i in range(l):
            sum += (ret[t].iloc[i] - data[t].iloc[i].values@gamma@f_list[t])**2
            ret_2 += ret[t].iloc[i]**2
    
    return 1 - sum/ret_2

def pred_R_squared(ret, data, gamma, f_list):

    lambda_t = np.mean(np.array(f_list), axis = 0)
    sum = 0
    ret_2 = 0
    l = len(ret[0])
    for t in range(len(data)):
        for i in range(l):
            sum += (ret[t].iloc[i] - data[t].iloc[i].values@gamma@lambda_t)**2
            ret_2 += ret[t].iloc[i]**2
    
    return 1 - sum/ret_2

def total_R_squared_kr(ret, Q, v):

    R = np.array(ret).flatten()

    return 1 - np.sum((R-Q@v)**2)/np.sum(R**2)

def total_R_squared_kr_out_os(ret, g_list, c_list):
    sum = 0
    ret_2 = 0
    l = len(ret[0])
    for t in range(len(g_list)):

        f_hat = g_list[t].T@(c_list[t].reshape(-1,1))

        for i in range(l): 

            sum += (ret[t].iloc[i] - (g_list[t]@f_hat)[i])**2
            ret_2 += ret[t].iloc[i]**2
    
    return 1 - sum/ret_2

def total_R_squared_kr_LR(ret, B, K, v, f_list):

    sum = 0
    ret_2 = 0
    v_mat = v.reshape(-1,5)
    
    for t in range(len(f_list)):
        pred = np.zeros((len(ret[0]), 1))
        for i in range(B.shape[1]):
            pred += (B.T@K[t*100:(t+1)*100,:].T)[i,:].reshape(-1,1)*v_mat[i,:].reshape(1,-1)@f_list[t].reshape(-1,1)
        sum += np.sum((ret[t+1].values - pred.flatten())**2)
        ret_2 += np.sum(ret[t+1].values**2)
        
    return 1-sum/ret_2
    