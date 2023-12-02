import numpy as np
import pandas as pd

def total_R_squared(ret, data, gamma, f_list):
    sum = 0
    ret_2 = 0
    l = len(ret[0])
    for t in range(len(data)-1):
        for i in range(l):
            sum += (ret[t+1].iloc[i] - data[t].iloc[i].values@gamma@f_list[t])**2
            ret_2 += ret[t+1].iloc[i]**2
    
    return 1 - sum/ret_2

def pred_R_squared(ret, data, gamma, f_list):

    lambda_t = np.mean(np.array(f_list), axis = 0)
    sum = 0
    ret_2 = 0
    l = len(ret[0])
    for t in range(len(data)-1):
        for i in range(l):
            sum += (ret[t+1].iloc[i] - data[t].iloc[i].values@gamma@lambda_t)**2
            ret_2 += ret[t+1].iloc[i]**2
    
    return 1 - sum/ret_2