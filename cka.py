import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# from 'scikit-learn pca code'
def centering(X):
    mean_vec = np.mean(X, axis=0)
    centered_X = X  - mean_vec
    return centered_X

def linear_kernel(X):
    return np.matmul(X, X.T)

# rbf_kernel: K(x,y) = exp(-gamma * ||x-y|^2), our gamma = 1/(2 * pow(sigma))
def rbf_HSIC(X, Y, sigma):
    n = X.shape[0]
    K = rbf_kernel(X, gamma = 1/(2 * pow(sigma, 2)))
    L = rbf_kernel(Y, gamma = 1/(2 * pow(sigma, 2)))

    K_center = centering(K)
    L_center = centering(L)
    
    rbf_hsic = np.matrix.trace(np.matmul(K_center, L_center)) / pow(n-1, 2)
    return rbf_hsic

def linear_HSIC(X, Y):
    n = X.shape[0]
    K = linear_kernel(X)
    L = linear_kernel(Y) 
    
    K_center = centering(K)
    L_center = centering(L)
    
    linear_hsic = np.matrix.trace(np.matmul(K_center, L_center)) / pow(n-1, 2)
    return linear_hsic

# sigma: fraction of the median distance between examples
def rbf_CKA(X, Y, sigma=None):
    hsic = rbf_HSIC(X, Y, sigma)
    hsic_K = rbf_HSIC(X, X, sigma)
    print(hsic_K)
    hsic_L = rbf_HSIC(Y, Y, sigma)
    print(hsic_L)
    return hsic / (np.sqrt(hsic_K * hsic_L))

# linear kernel & similar results -> similar resutls
def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    hsic_K = linear_HSIC(X, X)
    hsic_L = linear_HSIC(Y, Y)
    return hsic / (np.sqrt(hsic_K * hsic_L))

if __name__=='__main__':
    X = np.random.randn(100, 64)
    Y = np.random.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))