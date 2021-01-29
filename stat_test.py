import numpy as np
from numpy import transpose, trace, multiply, power, dot
from numpy.linalg import multi_dot, matrix_power
import scipy.stats as ss
from scipy.special import comb


def degree_test(A, level):
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
    # Average degree
    d_bar = (n-1)*alpha_n
    # Compute test statistic
    X_n = np.sum(np.power(d-d_bar,2))/((n-1)*alpha_n*(1-alpha_n))
    T = (X_n-n)/np.sqrt(2*n)
    # Asymptotic p-value
    pval = ss.norm.sf(abs(T), loc=0, scale=1)
    # Test result
    reject = (pval<=level/2)
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

