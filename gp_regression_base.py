import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from itertools import chain
import gaussian_elimination as ge
#import kernel_functions as kf

def kernel_wp_nonce_local(x, xp, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum
    
def cleverly_calculate_mu(K, Ks, y):
    #identity = np.eye(len(K))
    
    # Gaussian elimination on K
    #gaussian_elimination(K)
    
    # Same Gaussian elimination on identity
    #gaussian_elimination(identity)
    
    #return identity
    
    # Cheat by not being clever
    return Ks.T.dot(np.linalg.pinv(K)).dot(y);

            
## Load data
## We subsample the data, which gives us N pairs of (x, y)
data = loadmat('weather.mat')
x = np.arange(0, 1000, 20)
y = data['TMPMAX'][x]
x = x
y = y
N = len(y);

## Standardize data to have zero mean and unit variance
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## We want to predict values at x_* (denoted xs in the code)
M = 9
xs = np.linspace(np.min(x), np.max(x), M)

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1.0)**2

# Hyper parameter for kernel functions, to be optimized with max likelihood.
hyper_parameter = 1.0

## Compute covariance (aka "kernel") matrices
kernel = kernel_wp_nonce_local;
K = kernel(x, x, y, hyper_parameter) + sigma2*np.eye(N)
Ks = kernel(x, xs, y, hyper_parameter)
Kss = kernel(xs, xs, y, hyper_parameter)
 
## Compute conditional mean p(y_* | x, y, x_*)
#Kinv = np.linalg.pinv(K)
Kinv = ge.get_inverse(K)
mu = Ks.T.dot(Kinv).dot(y);
mu_test = cleverly_calculate_mu(K, Ks, y);
Sigma = Kss - Ks.T.dot(Kinv).dot(Ks);

## Plot the mean prediction
plt.figure(1)
plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
plt.plot(xs, mu_test) # mean prediction
plt.title('Mean prediction')
plt.show()

## Plot samples
#plt.figure(2)
#plt.plot(x, y, 'ko', markerfacecolor='k') # raw data
#S = 50 # number of samples
#samples = np.random.multivariate_normal(mu.reshape(-1), Sigma, S) # SxM
#for s in range(S):
#    plt.plot(xs, samples[s])
#plt.title('Samples')
#plt.show()

## Evaluate log-likelihood for a range of lambda's
#Q = 100
#possible_lambdas = np.linspace(1, 300, Q)
#loglikelihood = np.zeros(Q)
#for k in range(Q):
#    lambda_k = possible_lambdas[k]
#    K = kernel(x, x, lambda_k, theta) + sigma2*np.eye(N)
#    (_, logdet) = np.linalg.slogdet(K)
#    loglikelihood[k] = -0.5*N*np.log(2*np.pi) - 0.5*logdet - 0.5 * y.T.dot(np.linalg.pinv(K)).dot(y)
#
#idx = np.argmax(loglikelihood)
#lambda_opt = possible_lambdas[idx]
#plt.figure(3)
#plt.plot(possible_lambdas, loglikelihood)
#plt.plot(possible_lambdas[idx], loglikelihood[idx], '*')
#plt.title('Log-likelihood for \lambda');
#plt.xlabel('\lambda')
#plt.show()

