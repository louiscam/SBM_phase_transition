import numpy as np
from numpy import transpose, dot
from numpy.linalg import multi_dot


def sample_null_adj_mat(n, alpha):
    '''
    Function used to sample an adjacency matrix under the null distribution.
    Each off-diagonal entry of the matrix is sampled from a Bernoulli(alpha)
    distribution, and the diagonal enries are all set to 0 (no self-edge).
    
    Args:
      n: (int) number of nodes
      alpha: (float) edge probability
      
    Returns:
      An nxn adjacency matrix
    '''
    # Edge presence
    bern_vars = np.random.binomial(n=1, p=alpha, size=int(n*(n-1)/2))
    # Create upper triangular adjacency matrix
    tri = np.zeros((n,n))
    tri[np.triu_indices(n,1)] = bern_vars
    # Symmetrize adjacency matrix
    adj = tri+np.transpose(tri)
    return(adj)


def sample_alt_adj_matrix_sym(n, K, a, b, exact=True):
    '''
    Function used to sample a symmetric adjacency matrix under the alternative
    hypothesis with K communities. The matrix P is defined with all off-diagonal
    entries equal to b and diagonal entries equal to a. Membership vectors are 
    pure (n/K nodes per class) or sampled i.i.d. from a Multinomial(1/K,...,1/K).
    This allows to create the ideal matrix Omega, which is used to sample A with
    A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal elements, and A[i,i]=0 for 
    all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      a: (float) edge probability within a community
      b: (float) edge probability between distinct communities
      exact: (Boolean) if True, then there are exactly n/K pure nodes per community
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    P = (a-b)*np.identity(K)+b*np.ones(K)
    # Sample membership vectors
    if exact:
        #coms = np.random.permutation(np.array([np.repeat(i,int(n/K)) for i in range(K)]).flatten())
        coms = np.array([np.repeat(i,int(n/K)) for i in range(K)]).flatten()
        pi_mat = np.array([np.eye(1,K,i)[0] for i in coms])
    else:
        # pi_mat = np.random.multinomial(1, [1/K]*K, size=n)
        pi_mat = np.random.multinomial(1, [0.4,0.6], size=n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)


def sample_alt_adj_matrix_rank1(n, K, eta, c, exact=True):
    '''
    Function used to sample a symmetric adjacency matrix under the alternative
    hypothesis with K communities. The rank-1 matrix P is defined as c*eta*eta^T. 
    Membership vectors are pure (n/K nodes per class) or sampled i.i.d. from a 
    Multinomial(1/K,...,1/K). This allows to create the ideal matrix Omega, which is 
    used to sample A with A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal 
    elements, and A[i,i]=0 for all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      eta: (Numpy array) unique eigenvector of P
      c: (float) nonzero eigenvalue of P
      exact: (Boolean) if True, then there are exactly n/K pure nodes per community
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    P = c*dot(eta,transpose(eta))
    # Sample membership vectors
    if exact:
        #coms = np.random.permutation(np.array([np.repeat(i,int(n/K)) for i in range(K)]).flatten())
        coms = np.array([np.repeat(i,int(n/K)) for i in range(K)]).flatten()
        pi_mat = np.array([np.eye(1,K,i)[0] for i in coms])
    else:
        pi_mat = np.random.multinomial(1, [1/K]*K, size=n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)


def sample_alt_adj_matrix_sym2(n, K, a, b, c, exact):
    '''
    Function used to sample a symmetric adjacency matrix under the alternative
    hypothesis with K communities. The matrix P is defined with the block diagonal
    structure of section 3.3 of Jin et al (GC test). Membership vectors are 
    sampled from a Dirichlet distribution with equal probability for each class.
    This allows to create the ideal matrix Omega, which is used to sample A with
    A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal elements, and A[i,i]=0 for 
    all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      a: (float) edge probability within a community
      b: (float) edge probability between first group of communities
      c: (float) edge probability between second group of communities
      exact: (Boolean) if True, then there are exactly n/K pure nodes per community
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    B1 = (a-b)*np.identity(int(K/2))+b*np.ones(int(K/2))
    B2 = c*np.ones((int(K/2),int(K/2)))
    P = np.block([[B1,B2],[B2,B1]])
    # Sample membership vectors
    if exact:
        coms = np.random.permutation(np.array([np.repeat(i,int(n/K)) for i in range(K)]).flatten())
        pi_mat = np.array([np.eye(1,K,i)[0] for i in coms])
    else:
        pi_mat = np.random.multinomial(1, [1/K]*K, size=n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)


def sample_MMSBM_alt_adj_matrix_sym(n, K, a, b):
    '''
    Function used to sample a symmetric adjacency matrix under the alternative
    hypothesis with K communities. The matrix P is defined with all off-diagonal
    entries equal to b and diagonal entries equal to a. 
    Membership vectors are sampled from a Dirichlet distribution with equal 
    probability for each class. This allows to create the ideal matrix Omega, which 
    is used to sample A with A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal 
    elements, and A[i,i]=0 for all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      a: (float) edge probability within a community
      b: (float) edge probability between distinct communities
      
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    P = (a-b)*np.identity(K)+b*np.ones(K)
    # Sample membership vectors
    pi_mat = np.random.dirichlet(alpha = 0.1*np.ones(K), size = n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)


def sample_MMSBM_alt_adj_matrix_asym(n, K, a, b):
    '''
    Function used to sample an asymmetric adjacency matrix under the alternative
    hypothesis with K communities. The matrix P is defined with all off-diagonal
    entries sampled from Unif[b-e,b+e] where e<<b, and diagonal entries equal to a. 
    Membership vectors are sampled from a Dirichlet distribution with a given
    probability for each class. This allows to create the ideal matrix Omega, which 
    is used to sample A with A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal 
    elements, and A[i,i]=0 for all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      a: (float) edge probability within a community
      b: (float) edge probability between distinct communities
      
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    e = min(1-b,b)/6
    unif_vars = np.random.uniform(low=b-e, high=b+e, size=int(K*(K-1)/2))
    tri = np.zeros((K,K))
    tri[np.triu_indices(K,1)] = unif_vars
    P = tri+transpose(tri)+a*np.identity(K)
    # Sample membership vectors
    probs = np.random.uniform(0,10,K)
    pi_mat = np.random.dirichlet(alpha = probs, size = n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)


def sample_MMSBM_alt_adj_matrix_asym2(n, K, eta, c, probs):
    '''
    Function used to sample an asymmetric adjacency matrix under the alternative
    hypothesis with K communities. The rank-1 matrix P is defined as c*eta*eta^T. 
    Membership vectors are sampled from a Dirichlet distribution with a given
    probability for each class. This allows to create the ideal matrix Omega, which 
    is used to sample A with A[i,j]~Bernoulli(Omega[i,j]) for all off-diagonal 
    elements, and A[i,i]=0 for all diagonal elements (no self-edge).
    
    Args:
      n: (int) number of nodes
      K: (int) number of communities
      eta: (Numpy array) unique eigenvector of P
      c: (float) nonzero eigenvalue of P
      probs: (Numpy array) Dirichlet parameters for the mixed membership vectors
      
    Returns:
      An nxn adjacency matrix  
    '''
    # Construct P
    P = c*dot(eta,transpose(eta))
    # Sample membership vectors
    pi_mat = np.random.dirichlet(alpha = probs, size = n)
    # Construct Omega
    Omega = multi_dot([pi_mat, P, transpose(pi_mat)])
    # Sample adjacency matrix
    bern_vars = np.random.binomial(n=1, p=Omega)
    tri = np.triu(bern_vars, k=1)
    adj = tri+transpose(tri)
    return(adj)
