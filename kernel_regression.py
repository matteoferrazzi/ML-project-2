import numpy as np
import pandas as pd

def Kernel(x,xt, tk, l=1):
    #x is a 1x94 or a 100x94
    #xt is always 100x94

    #Linear
    if tk==0:
        return x@xt.T 

    #Gaussian
    if tk==1:
        K=np.zeros([x.shape[0], xt.shape[0]])
        for i in range(0,x.shape[0]):
            K[i]=np.exp(-np.sum((xt-x[i])**2, axis=1)/(2*l**2))
        return K
    
def K_LR(data2, tk, l):

    return Kernel(data2, data2, tk, l)
    
def solve_v_kernel (ret, lambda1, data, f_list, N, Omega, K):
    #f is a list
    #compute Q matrix
    T=len(data)
    A=np.array(f_list)@np.array(f_list).T
    Q=np.zeros([(T-1)*N, (T-1)*N])
    R=np.array(ret[1:]).flatten()

    for t in range(0,T-1):
        for s in range(0,T-1):
            Q[t*100:(t+1)*100,s*100:(s+1)*100]=A[t,s]*K[t*100:(t+1)*100,s*100:(s+1)*100]
    v=np.linalg.solve(Q+lambda1*Omega, R)

    return v, Q

def solve_g_kernel(data, f_list, v, t, K):

    T=len(data)

    g = np.sum([(K[t*100:(t+1)*100,s*100:(s+1)*100]@v[s*100:(s+1)*100]).reshape(-1,1)@f_list[s].reshape(1,-1) for s in range(0,T-1)], axis=0)#remember that f_list[s] is F_{s+1} 
    
    return g

def Gram_matrix(data,v, f_list, t, K):

    g=solve_g_kernel(data, f_list, v, t, K)

    return g@g.T

def solve_f(ret, v, lambda2, data, f_list, Omega, K):
    T=len(data)
    f_list_new=[]

    for t in range(0,T-1):
        G = Gram_matrix(data, v, f_list, t, K)
        c = np.linalg.solve(G+lambda2*Omega, ret[t+1])
        g = solve_g_kernel(data, f_list, v, t, K)
        f_list_new.append(g.T@c)
    f_list=f_list_new.copy()

    return f_list, g, G

def kernel_regression(data, ret, f_list, lambda1, lambda2, Omega1, Omega2, max_iter, N, K):
    
    for i in range(max_iter):
        print(i)
        v, Q = solve_v_kernel(ret, lambda1, data, f_list, N, Omega1, K)
        f_list, g, G = solve_f(ret, v, lambda2, data, f_list, Omega2, K)

    return f_list, v, Q, g, G

###### LOW RANK ######

def pivoted_chol(K, m_hat):

    d = np.diag(K).reshape(-1,1)
    p_max = np.argmax(d)
    d_max = d[p_max]

    e = np.zeros(len(K)).reshape(-1,1)
    e[p_max] = 1

    L = np.sqrt(1/d_max) * K@e
    B = np.sqrt(1/d_max) * np.eye(len(K))@e

    d = d-L*L

    for m in range(1,m_hat):
        print(m)
        p_max = np.argmax(d)
        d_max = d[p_max]

        e = np.zeros(len(K)).reshape(-1,1)
        e[p_max] = 1

        l = np.sqrt(1/d_max) * (K-L@L.T)@e
        b = np.sqrt(1/d_max) * (np.eye(len(K))-B@L.T)@e

        L = np.concatenate((L, l), axis = 1)
        B = np.concatenate((B, b), axis = 1)

        d = d-l*l

    return L,B

def solve_v_LR(data, B, ret, f_list, K, lambda1, Omega, m_hat):

    rhs = np.sum([np.kron(B.T@K[i*100:(i+1)*100,:].T@Omega@(B.T@K[i*100:(i+1)*100,:].T).T, 
                        f_list[i].reshape(-1,1)@f_list[i].reshape(1,-1)) for i in range(len(data)-1)], 
                        axis=0) + lambda1*np.eye(m_hat*len(f_list[0]))

    lhs = np.sum([np.kron(B.T@K[i*100:(i+1)*100,:].T@Omega,f_list[i].reshape((-1,1)))@ret[i+1] 
                for i in range(len(data)-1)], axis=0)

    v = np.linalg.solve(rhs, lhs)

    return v

def solve_g_kernel_LR(B, v, t, K, N):

    v_mat = v.reshape(-1,5)

    g = np.sum([(B.T@K[t*100:(t+1)*100,:].T)[i,:].reshape(-1,1)@v_mat[i,:].reshape(1,-1) for i in range(B.shape[1])], axis=0)

    return g

def gram_matrix_LR(B, v, t, K, N):

    g = solve_g_kernel_LR(B, v, t, K, N)

    return g@g.T

def solve_f_LR(ret, v, B, lambda2, data, Omega, K, N):
    
    T = len(data)
    f_list_new = []

    for t in range(T-1):
        G = gram_matrix_LR(B, v, t, K, N)
        g = solve_g_kernel_LR(B, v, t, K, N)
        c = np.linalg.solve(G+lambda2*Omega, ret[t+1])
        f_list_new.append(g.T@c)

    return f_list_new, G, g

def kernel_regression_LR(data, K, B, ret, f_list, lambda1, lambda2, Omega, max_iter, m_hat, N):

    for i in range(max_iter):
        print(i)
        v = solve_v_LR(data, B, ret, f_list, K, lambda1, Omega, m_hat)
        f_list, G, g = solve_f_LR(ret, v, B, lambda2, data, Omega, K, N)

    return f_list, v, G, g

    