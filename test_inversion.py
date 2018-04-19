import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist

def kernel_wp_nonce_local(x, xp, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum

def cleverly_calculate_Ky(K, y):
    #identity = np.eye(len(K))
    
    # Gaussian elimination on K
    #gaussian_elimination(K)
    
    # Same Gaussian elimination on identity
    #gaussian_elimination(identity)
    
    #return identity
    
    # Cheat by not being clever
    return np.linalg.pinv(K).dot(y);

## Load data
## We subsample the data, which gives us N pairs of (x, y)
data = loadmat('weather.mat')
x = np.arange(0, 1000, 20)
y = data['TMPMAX'][x]
x = x[0:8]
y = y[0:8]
N = len(y);

## Standardize data to have zero mean and unit variance
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## We want to predict values at x_* (denoted xs in the code)
M = 1000
xs = np.linspace(np.min(x), np.max(x), M)

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1.0)**2

# Hyper parameter for kernel functions, to be optimized with max likelihood.
hyper_parameter = 1.0

## Compute covariance (aka "kernel") matrices
K = kernel_wp_nonce_local(x, x, y, hyper_parameter) + sigma2*np.eye(N)
Ks = kernel_wp_nonce_local(x, xs, y, hyper_parameter)
Kss = kernel_wp_nonce_local(xs, xs, y, hyper_parameter)
 
## Compute conditional mean p(y_* | x, y, x_*)
Kinv = np.linalg.pinv(K)
mu = Ks.T.dot(Kinv).dot(y)
mu_test = Ks.T.dot(cleverly_calculate_Ky(K,y))
Sigma = Kss - Ks.T.dot(Kinv).dot(Ks);

plt.figure(1)
plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
plt.plot(xs, mu_test) # mean prediction
plt.title('Mean prediction')
plt.show()