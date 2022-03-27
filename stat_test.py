import numpy as np
from numpy import transpose, trace, multiply, power, dot
from numpy.linalg import multi_dot, matrix_power
import scipy.stats as ss
from scipy.special import comb
from sklearn.cluster import KMeans
from TracyWidom import TracyWidom
import itertools


def degree_test(A, level, two_sided=False):
    '''
    Function used to perform the degree-based chi-squared test for the global
    detection problem. The test statistic is computed as the scaled sum of squared
    discrepancies of node degrees with respect to the average degree. Properly 
    centered and scaled, this statistic asymptotically follows a standard normal
    distribution. This property is used to provide an asymptotic p-value for the 
    two-sided test. Rejection of the null hypothesis is determined using the test 
    level specified as input.
    
    Args:
      A: (Numpy array) adjacency matrix
      level: (float) level of the test
      two_sided: (Boolean) whether the test should be two-sided
      
    Returns:
      A dictionary containing the test statistic, asymptotic p-value and Boolean
      indicating whether the null hypothesis (only one community) was rejected.
    '''
    # Helpers
    n = A.shape[0]
    # Vector of degrees
    d = np.sum(A, axis=1, keepdims=True)
    # Edge probability estimate
    alpha_n = np.sum(A)/(n*(n-1))
    if (alpha_n==0):
        alpha_n = 2/(n*(n-1))
    if (alpha_n==1):
        alpha_n = (n*(n-1)-2)/(n*(n-1))
    # Average degree
    d_bar = (n-1)*alpha_n
    # Compute test statistic
    X_n = np.sum(np.power(d-d_bar,2))/((n-1)*alpha_n*(1-alpha_n))
    T = (X_n-n)/np.sqrt(2*n)
    # Asymptotic p-value and test result
    if (two_sided==True):
        pval = ss.norm.sf(abs(T), loc=0, scale=1)
        reject = (pval<=level/2)
    else:
        pval = ss.norm.sf(T, loc=0, scale=1)
        reject = (pval<=level)
        
    return({'test_stat': T, 'p_val': pval, 'reject': reject})


def ST_test(A, level):
    '''
    Function used to perform the Signed Triangle (ST) test, which counts the
    number of signed 3-cycles in the adjacency matrix. Properly centered and scaled, 
    this statistic asymptotically follows a standard normal distribution. This 
    property is used to provide an asymptotic p-value for the two-sided test. 
    Rejection of the null hypothesis is determined using the test level specified as 
    input.
    
    Args:
      A: (Numpy array) adjacency matrix
      level: (float) level of the test
      
    Returns:
      A dictionary containing the test statistic, an asymptotic p-value and a 
      logical indicating whether the null hypothesis (only one community) was 
      rejected.
    '''
    # Matrix dimension
    n = A.shape[0]
    # Edge probability estimate
    alpha_n = np.sum(A)/(n*(n-1))
    if (alpha_n==0):
        alpha_n = 2/(n*(n-1))
    if (alpha_n==1):
        alpha_n = (n*(n-1)-2)/(n*(n-1))
    # Helper quantities
    B = A-alpha_n
    B2 = dot(B,B)
    B3 = dot(B2,B)
    # Compute test statistic
    T_n = trace(B3)-3*trace(B*B2)+2*trace(B*B*B)
    T = T_n/np.sqrt(comb(n,3)*(alpha_n**3)*((1-alpha_n)**3))
    # Asymptotic p-value
    pval = ss.norm.sf(abs(T), loc=0, scale=1)
    # Test result
    reject = (pval<=level/2)
    return({'test_stat': T, 'p_val': pval, 'reject': reject})


def SQ_test(A, level):
    '''
    Function used to perform the Signed Quadrilateral (SQ) test, which counts the
    number of signed 4-cycles in the adjacency matrix. Properly centered and scaled, 
    this statistic asymptotically follows a standard normal distribution. This 
    property is used to provide an asymptotic p-value for the two-sided test. 
    Rejection of the null hypothesis is determined using the test level specified as 
    input.
    
    Args:
      A: (Numpy array) adjacency matrix
      level: (float) level of the test
      
    Returns:
      A dictionary containing the test statistic, an asymptotic p-value and a 
      logical indicating whether the null hypothesis (only one community) was 
      rejected.
    '''
    # Matrix dimension
    n = A.shape[0]
    # Edge probability estimate
    alpha_n = np.sum(A)/(n*(n-1))
    if (alpha_n==0):
        alpha_n = 2/(n*(n-1))
    if (alpha_n==1):
        alpha_n = (n*(n-1)-2)/(n*(n-1))
    # Helper quantities
    B = A-alpha_n
    D = np.diag(np.diag(B))
    B2 = dot(B,B)
    B3 = dot(B2,B)
    B4 = dot(B3,B)
    hB2 = B*B
    hB4 = hB2*hB2
    # Compute partial sums
    S1 = trace(B*B3)-2*trace(hB2*B2)-np.sum(multi_dot([D,hB2,D]))+2*trace(hB4)
    S2 = trace(B2*B2)-2*trace(hB2*B2)-np.sum(hB4)+2*trace(hB4)
    S3 = trace(hB2*B2)-trace(hB4)
    S4 = np.sum(hB4)-trace(hB4)
    S5 = np.sum(multi_dot([D,hB2,D]))-trace(hB4)
    S6 = trace(hB4)
    # Compute test statistic
    Q_n = trace(B4)-4*S1-2*S2-4*S3-S4-2*S5-S6
    T = Q_n/(2*np.sqrt(2)*np.power(n,2)*np.power(alpha_n,2)*np.power(1-alpha_n,2))
    # Asymptotic p-value
    pval = ss.norm.sf(abs(T), loc=0, scale=1)
    # Test result
    reject = (pval<=level/2)
    return({'test_stat': T, 'p_val': pval, 'reject': reject})


def PET_test(A, level):
    '''
    Function used to perform the Power-Enhanced Test (PET), which combines the
    strengths of the degree-based chi-square test and the SQ test. The test 
    statistic is equal to the sum of the squared degree test statistic and the 
    squared SQ test statistic. The PET statistic follows a Chi-Squared(2) 
    distribution asymptotically. This property is used to provide an asymptotic p-
    value for PET. Rejection of the null hypothesis is determined using the test 
    level specified as input.
    
    Args:
      A: (Numpy array) adjacency matrix
      level: (float) level of the test
      
    Returns:
      A dictionary containing the test statistic, an asymptotic p-value and a 
      logical indicating whether the null hypothesis (only one community) was 
      rejected.
    '''
    # Compute test statistic
    T_X = degree_test(A, level)['test_stat']
    T_Q = SQ_test(A, level)['test_stat']
    T = T_X**2+T_Q**2
    # Asymptotic p-value
    pval = ss.chi2.sf(x=T, df=2, loc=0, scale=1)
    # Test result
    reject = (pval<=level)
    return({'test_stat': T, 'p_val': pval, 'reject': reject})


def lei_test(A, K, level):
    '''
    Function used to perform the eigenvalue-based test of Lei (2016). 
    The test statistic uses the largest singular value of a residual matrix 
    obtained by subtracting the estimated block mean effect of the adjacency matrix
    Args:
      A: (Numpy array) adjacency matrix
      K: (int) number of communities in the null SBM model
      level: (float) level of the test
      
    Returns:
      A dictionary containing the test statistic, asymptotic p-value and Boolean
      indicating whether the null hypothesis (only one community) was rejected.
    '''
    # Helpers
    n = A.shape[0]
    
    # Estimate community membership with spectral clustering
    w, v = np.linalg.eigh(A)
    magnitude_ordering = np.argsort(-np.abs(w))
    w = w[magnitude_ordering]
    v = v.T[magnitude_ordering].T
    v_selected = v[:,:K]
    kmeans = KMeans(n_clusters=K, random_state=0).fit(v_selected)
    z_hat = kmeans.labels_
    pi_mat_hat = np.array([np.eye(1,K,c)[0] for c in z_hat])
    
    # Compute residual matrix A_tilde
    N_hat = {k: np.where(z_hat==k)[0] for k in range(K)}
    n_hat = {k: np.sum(z_hat==k) for k in range(K)}
    P_hat = np.zeros((K,K))
    for k,l in itertools.combinations_with_replacement(range(K),2):
        if (k!=l):
            P_hat[k,l] = np.sum(A[np.ix_(N_hat[k],N_hat[l])])/(n_hat[k]*n_hat[l])
            P_hat[l,k] = P_hat[k,l]
        else:
            P_hat[k,k] = np.sum(A[np.ix_(N_hat[k],N_hat[k])])/(n_hat[k]*(n_hat[k]-1))
    Omega_hat = multi_dot([pi_mat_hat, P_hat, transpose(pi_mat_hat)])
    A_tilde = (A-Omega_hat)/np.sqrt((n-1)*Omega_hat*(1-Omega_hat))
    np.fill_diagonal(A_tilde, 0)
    
    # Compute test statistic
    w_tilde, _ = np.linalg.eigh(A_tilde)
    sig1 = np.max(np.abs(w_tilde))
    T = (n**(2/3))*(sig1-2)
    tw1 = TracyWidom(beta=1)
    pval = 1-tw1.cdf(T)
    reject = (pval<level/2)

    return({'test_stat': T, 'p_val': pval, 'reject': reject})


def wangbickel_likelihood_method(A, Kmax, tol=1e-8, tol_fp=1e-4, max_iter=100, init='random', seed=13):
    '''
    Function to compute the model selection criterion of Wang & Bickel (2016)
    Args:
      A: (Numpy array) adjacency matrix
      Kmax: (int) maximum value of K in Wang & Bickel (2016)
      tol: (float) tolerance level to stop the EM algorithm
      tol_fp: (float) tolerance level to stop the fixed point iterations
      max_iter: (int) maximum number of iterations
      init: (str) initialization strategy for tau, one of 'spectral' or 'random'
      seed: (int) seed for initialization of the EM algorithm
      
    Returns:
      Value of the model selection criterion
    '''
    # Optimize variational log-likelihood
    n = A.shape[0]
    J = np.array([optimize_varloglik(A, k+1, tol=tol, tol_fp=tol_fp, max_iter=max_iter, init=init, seed=13)
                  for k in range(Kmax)])
    lambda_list = np.arange(0, 0.3, 1e-3)
    entropy = -np.inf
    for lambd in lambda_list:
        betas = np.array([beta(k+1, J[k], n, lambd, seed=13) for k in range(Kmax)])
        w = betas/np.sum(betas)
        entropy_next = -np.sum(w*np.log(w))
        if (entropy_next > entropy):
            lambd_star = lambd
            beta_star = betas
            entropy = entropy_next
    betas_dict = {f'beta_{k+1}': beta_star[k] for k in range(Kmax)}
    K_best = np.argmax(beta_star)+1
    reject = (K_best==2) # (K_best>1)
    return({'betas': betas_dict, 'reject': reject})


def varloglik(tau, P, a, A):
    '''
    Function to compute the variational log-likelihood of the data
    Args:
      tau: (Numpy array) variational parameters (parameter of a multinomial for the K clusters)
      P: (Numpy array) community mixing matrix
      a: (Numpy array) multinomial parameters of the SBM cluster assignment
      A: (Numpy array) adjacency matrix
    Returns:
      The value of the variational log-likelihood
    '''
    n = A.shape[0]
    M = np.ones((n,n))
    np.fill_diagonal(M,0)
    J = np.sum(np.log(a)*np.sum(tau, axis=0))
    J += -np.sum(tau*np.log(tau))
    J += 0.5*np.sum(multi_dot([tau, np.log(P), tau.T])*A*M)
    J += 0.5*np.sum(multi_dot([tau, np.log(1-P), tau.T])*(1-A)*M)
    return J


def ff(x, a, P, A):
    '''
    Function used to solve for tau (parameter of a multinomial for the K clusters)
    using a fixed point iteratio algorithm
    Args:
      x: (Numpy array) input, corresponds to the current estimate of tau
      a: (Numpy array) multinomial parameters of the SBM cluster assignment
      P: (Numpy array) community mixing matrix
      A: (Numpy array) adjacency matrix
      
    Returns:
      The updated value of tau
    '''
    n = A.shape[0]
    M = np.ones((n,n))
    np.fill_diagonal(M,0)
    log_y = multi_dot([A, x, np.log(P)])
    log_y += multi_dot([M*(1-A), x, np.log(1-P)])
    log_y += np.log(a)
    log_y += -np.mean(log_y)
    y = np.exp(log_y)
    y = y/np.sum(y, axis=1)[:,np.newaxis]
    return y


def fixed_point_method(ff, x, a, P, A, tol=1e-4):
    '''
    Function to run fixed point iteration algorithm to the function ff, i.e.
    find the fixed point x* such that x* = ff(x*)
    Args:
      ff: (function) function for fixed point interation
      x: (Numpy array) input, corresponds to the current estimate of tau
      a: (Numpy array) multinomial parameters of the SBM cluster assignment
      P: (Numpy array) community mixing matrix
      A: (Numpy array) adjacency matrix
      tol: (float) tolerance level to stop the algorithm
      
    Returns:
      The fixed point of ff
    '''
    c = 1
    while (np.sum(np.abs(x-ff(x, a, P, A))) > tol):
        x = ff(x, a, P, A)
        c += 1
        if (c>5000):
            # print('tau failed to converge')
            break
    return x


def optimize_varloglik(A, KK, tol=1e-8, tol_fp=1e-4, max_iter=100, init='random', seed=13):
    '''
    Function to optimize the variational log-likelihood using variational EM 
    (from Daudin, Picard and Robin, 2006)
    Args:
      A: (Numpy array) adjacency matrix
      KK: (int) the desired number of communities
      tol: (float) tolerance level to stop the EM algorithm
      tol_fp: (float) tolerance level to stop the fixed point iterations
      max_iter: (int) maximum number of iterations
      init: (str) initialization strategy for tau, one of 'spectral' or 'random'
      seed: (int) seed for initialization of the EM algorithm
      
    Returns:
      The maximized variational log-likelihood
    '''
    
    n = A.shape[0]
    
    if (KK>1):
        # For tau_init use spectral clustering on the Laplacian
        eps = 1e-8
        
        if (init=='spectral'):
            D_invsqrt = np.diag((np.sum(A+eps, axis=1)**(-1/2)))
            L = multi_dot([D_invsqrt,A+eps,D_invsqrt])
            w, v = np.linalg.eigh(L)
            magnitude_ordering = np.argsort(-np.abs(w))
            w = w[magnitude_ordering]
            v = v.T[magnitude_ordering].T
            v_selected = v[:,:KK]
            kmeans = KMeans(n_clusters=KK, random_state=0).fit(v_selected)
            z_init = kmeans.labels_
            tau_init = np.array([np.eye(1,KK,c)[0] for c in z_init])
#             tau_init_clipped = np.clip(tau_init, a_min=0.1, a_max=None)
#             tau_init = tau_init_clipped/np.sum(tau_init_clipped)
        # Or initialize based on random
        if (init=='random'):
            np.random.seed(seed)
            tau_init = np.random.uniform(0,1,(n,KK))
            tau_init = np.divide(tau_init, np.sum(tau_init, axis=1)[:, np.newaxis])
        

        # EM algorithm parameters
        n = A.shape[0]
        M = np.ones((n,n))
        np.fill_diagonal(M,0)
        clip_val = 1e-8

        # Start with initial guess for tau, a and P
        a_new = np.mean(tau_init, axis=0)
        a_new_clipped = np.clip(a_new, a_min=clip_val, a_max=None)
        a_new = a_new_clipped/np.sum(a_new_clipped)
        P_new = multi_dot([tau_init.T, A, tau_init])/multi_dot([tau_init.T, M, tau_init])
        P_new = np.clip(P_new, a_min=clip_val, a_max=1-clip_val)
        tau_new = fixed_point_method(ff, tau_init, a_new, P_new, A, tol=tol_fp)

        delta_theta = np.inf
        c = 1
        while (delta_theta>tol):

            # Store previous values
            a_old = a_new
            P_old = P_new
            tau_old = tau_new

            # Update a_new, P_new, tau_new
            a_new = np.mean(tau_old, axis=0)
            a_new_clipped = np.clip(a_new, a_min=clip_val, a_max=None)
            a_new = a_new_clipped/np.sum(a_new_clipped)
            P_new = multi_dot([tau_old.T, A, tau_old])/multi_dot([tau_old.T, M, tau_old])
            P_new = np.clip(P_new, a_min=clip_val, a_max=1-clip_val)
            tau_new = fixed_point_method(ff, tau_old, a_new, P_new, A, tol=tol_fp)

            # Assess convergence
            c += 1
            delta_theta = np.sum(np.abs(P_new-P_old))+np.sum(np.abs(a_new-a_old))
            if (c>max_iter):
                break

        tau_new_clipped = np.clip(tau_new, a_min=clip_val, a_max=None)
        tau_new = np.divide(tau_new_clipped, np.sum(tau_new_clipped, axis=1)[:,np.newaxis])
    
    else:
        tau_new = np.ones((n,1))
        a_new = np.array([1])
        P_new = np.array([[np.sum(A)/(n*(n-1))]])
    
    J = varloglik(tau_new, P_new, a_new, A)
    
    return J


def beta(KK, J, n, lambd, seed=13):
    '''
    Function to compute the model selection criterion of Wang & Bickel (2016)
    Args:
      A: (Numpy array) adjacency matrix
      seed: (int) seed for initialization of the EM algorithm
      
    Returns:
      Value of the model selection criterion
    '''
    
    b = J-lambd*KK*(1+KK)*n*np.log(n)/2
    return b






# def wangbickel_likelihood_method(A, Kmax, tol=1e-8, tol_fp=1e-10, max_iter=6000, seed=13):
#     '''
#     Function to compute the model selection criterion of Wang & Bickel (2016)
#     Args:
#       A: (Numpy array) adjacency matrix
#       Kmax: (int) maximum value of K in Wang & Bickel (2016)
#       tol: (float) tolerance level to stop the EM algorithm
#       tol_fp: (float) tolerance level to stop the fixed point iterations
#       max_iter: (int) maximum number of iterations
#       seed: (int) seed for initialization of the EM algorithm
      
#     Returns:
#       Value of the model selection criterion
#     '''
#     # Optimize variational log-likelihood
#     n = A.shape[0]
#     J = np.array([optimize_varloglik(A, k+1, tol=tol, tol_fp=tol_fp, max_iter=max_iter, seed=seed) for k in range(Kmax)]) 
#     lambda_list = np.array([0]) # np.arange(0, 0.3, 1e-3)
#     entropy = -np.inf
#     for lambd in lambda_list:
#         betas = np.array([beta(k+1, J[k], n, lambd, seed=13) for k in range(Kmax)])
#         w = betas/np.sum(betas)
#         entropy_next = -np.sum(w*np.log(w))
#         if (entropy_next > entropy):
#             lambd_star = lambd
#             beta_star = betas
#             entropy = entropy_next
#     betas_dict = {f'beta_{k+1}': beta_star[k] for k in range(Kmax)}
#     K_best = np.argmax(beta_star)+1
#     reject = (K_best==2) # (K_best>1)
#     return({'betas': betas_dict, 'reject': reject})
        

# def varloglik(tau, P, a, A):
#     '''
#     Function to compute the variational log-likelihood of the data
#     Args:
#       tau: (Numpy array) variational parameters (parameter of a multinomial for the K clusters)
#       P: (Numpy array) community mixing matrix
#       a: (Numpy array) multinomial parameters of the SBM cluster assignment
#       A: (Numpy array) adjacency matrix
#     Returns:
#       The value of the variational log-likelihood
#     '''
#     n = A.shape[0]
#     M = np.ones((n,n))
#     np.fill_diagonal(M,0)
#     J = np.sum(np.log(a)*np.sum(tau, axis=0))
#     J += -np.sum(tau*np.log(tau))
#     J += 0.5*np.sum(multi_dot([tau, np.log(P), tau.T])*A*M)
#     J += 0.5*np.sum(multi_dot([tau, np.log(1-P), tau.T])*(1-A)*M)
#     return J


# def ff(x, a, P, A):
#     '''
#     Function used to solve for tau (parameter of a multinomial for the K clusters)
#     using a fixed point iteratio algorithm
#     Args:
#       x: (Numpy array) input, corresponds to the current estimate of tau
#       a: (Numpy array) multinomial parameters of the SBM cluster assignment
#       P: (Numpy array) community mixing matrix
#       A: (Numpy array) adjacency matrix
      
#     Returns:
#       The updated value of tau
#     '''
#     n = A.shape[0]
#     M = np.ones((n,n))
#     np.fill_diagonal(M,0)
#     log_y = multi_dot([A, x, np.log(P)])
#     log_y += multi_dot([M*(1-A), x, np.log(1-P)])
#     log_y += np.log(a)
#     log_y += -np.mean(log_y)
#     y = np.exp(log_y)
#     y = y/np.sum(y, axis=1)[:,np.newaxis]
#     return y


# def fixed_point_method(ff, x, a, P, A, tol=1e-10):
#     '''
#     Function to run fixed point iteration algorithm to the function ff, i.e.
#     find the fixed point x* such that x* = ff(x*)
#     Args:
#       ff: (function) function for fixed point interation
#       x: (Numpy array) input, corresponds to the current estimate of tau
#       a: (Numpy array) multinomial parameters of the SBM cluster assignment
#       P: (Numpy array) community mixing matrix
#       A: (Numpy array) adjacency matrix
#       tol: (float) tolerance level to stop the algorithm
      
#     Returns:
#       The fixed point of ff
#     '''
#     while (np.linalg.norm(x-ff(x, a, P, A)) > tol*np.linalg.norm(x)):
#         x = ff(x, a, P, A)
#     return x


# def optimize_varloglik(A, KK, tol=1e-8, tol_fp=1e-10, max_iter=6000, seed=13):
#     '''
#     Function to optimize the variational log-likelihood using variational EM 
#     (from Daudin, Picard and Robin, 2006)
#     Args:
#       A: (Numpy array) adjacency matrix
#       KK: (int) the desired number of communities
#       tol: (float) tolerance level to stop the EM algorithm
#       tol_fp: (float) tolerance level to stop the fixed point iterations
#       max_iter: (int) maximum number of iterations
#       seed: (int) seed for initialization of the EM algorithm
      
#     Returns:
#       The maximized variational log-likelihood
#     '''
    
#     # EM algorithm parameters
#     n = A.shape[0]
#     M = np.ones((n,n))
#     np.fill_diagonal(M,0)
#     flag = True
    
#     # Start with initial guess for tau
#     np.random.seed(seed)
#     c = 0
#     tau_hat = np.random.uniform(0,1,(n,KK))
#     tau_hat = np.divide(tau_hat, np.sum(tau_hat, axis=1)[:, np.newaxis])
#     J = -np.inf
    
#     while flag:
    
#         # Update a_hat and P_hat
#         a_hat = np.mean(tau_hat, axis=0)
#         P_hat = np.zeros((KK,KK))
#         for k,l in itertools.combinations_with_replacement(range(KK),2):
#             P_hat[k,l] = multi_dot([tau_hat[:,k, np.newaxis].T, A , tau_hat[:,l, np.newaxis]])[0][0]
#             P_hat[k,l] = P_hat[k,l]/multi_dot([tau_hat[:,k, np.newaxis].T, M , tau_hat[:,l, np.newaxis]])[0][0]
#             if (k!=l):
#                 P_hat[l,k] = P_hat[k,l]

#         # Update tau_hat
#         tau_hat = fixed_point_method(ff, tau_hat, a_hat, P_hat, A, tol=tol_fp)
#         J_new = varloglik(tau_hat, P_hat, a_hat, A)
#         c += 1

#         # Assess convergence
#         if (J_new-J>=tol*np.abs(J)) & (c<max_iter):
#             J = J_new
#         else:
#             flag = False
    
#     return J


# def beta(KK, J, n, lambd, seed=13):
#     '''
#     Function to compute the model selection criterion of Wang & Bickel (2016)
#     Args:
#       A: (Numpy array) adjacency matrix
#       seed: (int) seed for initialization of the EM algorithm
      
#     Returns:
#       Value of the model selection criterion
#     '''
    
#     b = J-lambd*KK*(1+KK)*n*np.log(n)/2
#     return b
