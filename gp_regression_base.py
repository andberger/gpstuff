import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist

####### HELPER FUNCTIONS #######

# K = kernel(x, y, _lambda, _theta)
#   Evaluate the squared exponential kernel function with parameters
#   lambda and theta.
#
#   x and y should be NxD and MxD matrices. The resulting
#   covariance matrix will be of size NxM.
#def kernel(x, y, _lambda, _theta):
#    D2 = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean') # pair-wise distances, size: NxM
#    K = _theta * np.exp(-0.5 * D2 * _lambda) # NxM
#    return K

def kernel_wp_nonce(x, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return sigma * minimum

def kernel_wp_once(x, y, sigma):
    dist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'euclidean')
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return sigma * ( (np.power(minimum,3) / 3) + dist*(np.power(minimum,2) / 2) )

def kernel_wp_once2(x, y, sigma):

    # x er vigur 1xN
    # y er vigur 1xM
    # sigma er rauntala
    # viljum skila samfylgnifylki
    N = len(x)
    M = len(y)
    covmat = np.zeros((N, M))
    for i in range(N):
        a = x[i]
        for j in range(M):
            b = y[j]
            d1 = np.power(np.fmin(a, b), 3)/3
            d2 = np.power(np.fmin(a, b), 2)/2
            v = sigma*sigma*(d1 + np.abs(a-b) * d2)
            #print(N, M, i, j, v)
            covmat[i,j] = v
            
    return covmat
    
## Load data
## We subsample the data, which gives us N pairs of (x, y)
data = loadmat('weather.mat')
x = np.arange(0, 1000, 20)
y = data['TMPMAX'][x]
N = len(y);

## Standardize data to have zero mean and unit variance
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## We want to predict values at x_* (denoted xs in the code)
M = 1000
xs = np.linspace(np.min(x), np.max(x), M)

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1.0)**2

## Compute covariance (aka "kernel") matrices
K = kernel_wp_once(x, x, sigma2) + sigma2*np.eye(N) 
Ks = kernel_wp_once(x, xs, sigma2)
Kss = kernel_wp_once(xs, xs, sigma2)

 
## Compute conditional mean p(y_* | x, y, x_*)
Kinv = np.linalg.pinv(K)
mu = Ks.T.dot(Kinv).dot(y);
Sigma = Kss - Ks.T.dot(Kinv).dot(Ks);

## Plot the mean prediction
plt.figure(1)
plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
plt.plot(xs, mu) # mean prediction
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

