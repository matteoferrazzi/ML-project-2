import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import importlib
import download_clean_data as dc
import ipca
import metrics
import kernel_regression as kr
importlib.reload(dc) 
importlib.reload(ipca)
importlib.reload(metrics)
importlib.reload(kr)

def split_dataset(x,y, trsh):

    n = int(np.floor(len(y)*trsh))

    x_train=x[:n]
    y_train=y[:n]
    x_test=x[n:]
    y_test=y[n:]
        
    return x_train,y_train,x_test,y_test

def val_IPCA(y,x, trsh, gamma_first, max_iter):

    total_R2_dict = {}
    pred_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)

    gamma, _ = ipca.ipca(xx_train, yy_train, gamma_first.copy(), max_iter)
    print('done')

    yy_pred = []

    for i in range(len(xx_test)):

        f = ipca.solve_f(yy_test, xx_test, gamma, i)
        yy_pred.append(f)

    total_R2_dict[('IPCA')] = metrics.total_R_squared(yy_test, xx_test, gamma, yy_pred)
    pred_R2_dict[('IPCA')] = metrics.pred_R_squared(yy_test, xx_test, gamma, yy_pred)

    return total_R2_dict, pred_R2_dict

def val_IPCA_reg(y,x, trsh, lambda1_v, lambda2_v, gamma_first, max_iter, W_list):

    total_R2_dict = {}
    pred_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)

    for lambda1 in lambda1_v:

        for lambda2 in lambda2_v:

            gamma_reg_w, _ = ipca.ipca_reg_w(xx_train, yy_train, gamma_first.copy(), max_iter, lambda1, lambda2, W_list)

            yy_pred = []

            for i in range(len(xx_test)):

                f = ipca.solve_f_reg_w(yy_test, xx_test, gamma_reg_w, i, lambda1, W_list[i])
                yy_pred.append(f)

            total_R2_dict[('IPCA_reg', lambda1, lambda2)] = metrics.total_R_squared(yy_test, xx_test, gamma_reg_w, yy_pred)
            
    return total_R2_dict

def val_gaussian(y,x, trsh, lambda1_v, lambda2_v, l_v, N, f_list_input, Omega2, max_iter):

    total_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)

    for l in l_v:

        data2_train = xx_train.copy()
        data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
        data2_test = xx_test.copy()
        data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
        K_train = kr.K_LR(data2_train, 1, l)
        K_test_train = kr.Kernel(data2_test, data2_train, 1, l)
        
        for lambda1 in lambda1_v:

            for lambda2 in lambda2_v:
                
                f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)

                yy_pred = []
                g_list = []
                c_list = []
    
                for t in range(0,len(xx_test)):

                    g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                    G = g@g.T

                    g_list.append(g)

                    c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])

                    c_list.append(c)

                    yy_pred.append(g.T@c)

                total_R2_dict[('Gaussian', lambda1, lambda2, l)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict

def val_rq(y,x, trsh, lambda1_v, lambda2_v, l_v, alphas_v, N, f_list_input, Omega2, max_iter):

    total_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)

    for alpha in alphas_v:

        for l in l_v:

            data2_train = xx_train.copy()
            data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
            data2_test = xx_test.copy()
            data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
            K_train = kr.K_LR(data2_train, 2, l, alpha)
            K_test_train = kr.Kernel(data2_test, data2_train, 2, l, alpha)
            
            for lambda1 in lambda1_v:

                for lambda2 in lambda2_v:

                    f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)

                    yy_pred = []
                    g_list = []
                    c_list = []
        
                    for t in range(0,len(xx_test)):

                        g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                        G = g@g.T

                        g_list.append(g)

                        c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])

                        c_list.append(c)

                        yy_pred.append(g.T@c)

                    total_R2_dict[('Rational Quadratic', lambda1, lambda2, l, alpha)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict

def val_linear(y,x, trsh, lambda1_v, lambda2_v, N, f_list_input, Omega2, max_iter):

    total_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)
    Omega_train_inv = np.eye(len(xx_train)*N)

    data2_train = xx_train.copy()
    data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
    data2_test = xx_test.copy()
    data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
    K_train = kr.K_LR(data2_train, 0, 1)
    K_test_train = kr.Kernel(data2_test, data2_train, 0, 1)
    print('start cycle')

    g_list = []
    

    for lambda1 in lambda1_v:
        for lambda2 in lambda2_v:
            f_list, v, _, _, _ = kr.kernel_regression(xx_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega_train_inv, Omega2, max_iter, N, K_train)

            yy_pred = []

            g_list = []
            c_list = []
            for t in range(0,len(xx_test)):

                g = kr.solve_g_kernel(xx_train, f_list, v, t, K_test_train)
                G = g@g.T

                g_list.append(g)

                c = np.linalg.solve(G+lambda2*Omega2, yy_test[t])

                c_list.append(c)

                yy_pred.append(g.T@c)

            total_R2_dict[('Linear', lambda1, lambda2)] = metrics.total_R_squared_kr_out_os(yy_test, g_list, c_list)
                
    return total_R2_dict


def cross_val_gaussian_LR(y,x, trsh, lambda1_v, lambda2_v, alphas_v, N, f_list_input, Omega2, max_iter, m_hat):

    total_R2_dict = {}
    
    xx_train,yy_train,xx_test,yy_test = split_dataset(x,y, trsh)

    for alpha in alphas_v:

        data2_train = xx_train.copy()
        data2_train = np.array(np.array(data2_train).reshape(len(xx_train)*N,94)) #flatten data, build X
        data2_test = xx_test.copy()
        data2_test = np.array(np.array(data2_test).reshape(len(xx_test)*N,94))
        K_train = kr.K_LR(data2_train, 1, alpha)
        K_test = kr.K_LR(data2_test, 1, alpha)

        L_train, B_train = kr.pivoted_chol(K_train, m_hat)
        L_test, B_test = kr.pivoted_chol(K_test, m_hat)

        print(np.linalg.norm(L_test@L_test.T-K_test))

        for lambda1 in lambda1_v:
            for lambda2 in lambda2_v:
                f_list, v, G, g = kr.kernel_regression_LR(xx_train, K_train, B_train, yy_train, f_list_input.copy(), lambda1, lambda2, Omega2, max_iter, m_hat, N)

                yy_pred = []
                for t in range(0,len(xx_test)-1):

                    c = np.linalg.solve(G+lambda2*Omega2, yy_test[t+1])
                    yy_pred.append(g.T@c)

                vt = kr.solve_v_LR(xx_test, B_test, yy_test, yy_pred, K_test, lambda1, Omega2, m_hat)

                print(np.sum(B_test>0)/(B_test.shape[0]*B_test.shape[1]))

                total_R2_dict[('Gaussian', lambda1, lambda2, alpha)] = metrics.total_R_squared_kr_LR(yy_test, B_test, K_test, vt, yy_pred)
                
    return total_R2_dict

