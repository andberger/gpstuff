#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from itertools import chain
#import kernel_functions as kf

def kernel_wp_nonce_local(x, xp, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum
    
def squared_means_kernel(x, y, _lambda, _theta):
  D2 = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'sqeuclidean') # pair-wise distances, size: NxM
  K = _theta * np.exp(-0.5 * D2 * _lambda) # NxM
  return K

def cleverly_calculate_mu(K, Ks, y):
    #identity = np.eye(len(K))
    
    # Gaussian elimination on K
    #gaussian_elimination(K)
    
    # Same Gaussian elimination on identity
    #gaussian_elimination(identity)
    
    #return identity
    
    # Cheat by not being clever
    return Ks.T.dot(np.linalg.pinv(K)).dot(y);

def gaussian_elimination(A):
    eps = 0.00000000000000001;
    h = 0
    k = 0
    m = len(A[0])
    n = len(A)
    while h <= m and k <= n:
        i_max = h
        v_max = 0
        for i in range(h, m):
            v = abs(A[i, k])
            if v > v_max:
                i_max = i
                v_max = v
        
        if A[i_max, k] <= eps:
            k = k+1
        else:
            swap_rows(h, i_max, A)
            for i in range(h+1, m):
                f = A[i, k] / A[h, k]
                A[i, k] = 0
                for j in range(k+1,n):
                    A[i, j] = A[i, j] - A[k, j] * f
            h = h+1
            k = k+1
            
def swap_rows(i1, i2, A):
    A[i1], A[i2] = A[i2], A[i1]
            
#tt = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
#gaussian_elimination(tt)

## Load data
## We subsample the data, which gives us N pairs of (x, y)
data = loadmat('weather.mat')
x = np.arange(0, 1000, 20)
y = data['TMPMAX'][x]
#x = x[0:8]
#y = y[0:8]
N = len(y);

## Standardize data to have zero mean and unit variance
xmean = np.mean(x)
ymean = np.mean(y)
xstdev = np.std(x)
ystdev = np.std(y)
x = (x - xmean) / xstdev
y = (y - ymean) / ystdev

## We want to predict values at x_* (denoted xs in the code)
M = 100
xs = np.linspace(np.min(x), np.max(x), M)

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (0.9)**2

# Hyper parameter for kernel functions, to be optimized with max likelihood.
hyper_parameter = 1.0

lam = 13.456375838926174
theta = 1.0
## Compute covariance (aka "kernel") matrices
kernel = squared_means_kernel;
K = kernel(x, x, lam, theta) + sigma2*np.eye(N)
Ks = kernel(x, xs, lam, theta)
Kss = kernel(xs, xs, lam, theta)
 
## Compute conditional mean p(y_* | x, y, x_*)
Kinv = np.linalg.pinv(K)
mu = Ks.T.dot(Kinv).dot(y);
mu_test = cleverly_calculate_mu(K, Ks, y);
Sigma = Kss - Ks.T.dot(Kinv).dot(Ks);

## Plot the mean prediction
#plt.subplot(1, 3, 1)
#plt.figure(1)
#plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
#plt.plot(xs, mu_test) # mean prediction
#plt.title('Mean prediction')
#plt.show()

## Plot the mean prediction
#plt.subplot(1, 3, 2)
#plt.figure(1)
#plt.plot(x, y, 'o', markerfacecolor='y') # raw data
#plt.plot(xs, mu_test) # mean prediction
#plt.title('Mean prediction')
#plt.show()
#


## Plot samples
plt.subplot(1, 2, 2)

xdenorm = (x+xmean)*xstdev
ydenorm = (y+ymean)*ystdev

plt.figure(1)
ax = plt.gca()
ax.grid(color='0.7', linestyle='-', linewidth=1)
S = 50 # number of samples
samples = np.random.multivariate_normal(mu.reshape(-1), Sigma, S) # SxM
#for s in range(S):
#    plt.plot(xs, samples[s])

plt.xlabel('Time')
plt.ylabel('Temperature')
a = mu.reshape(-1)+Sigma.diagonal()
b = mu.reshape(-1)-Sigma.diagonal()
#plt.plot(xs, a, 'b-')
plt.plot(xs, mu, 'k-')
#plt.plot(xs, b, 'b-')

ax.fill_between(xs, a, b, facecolor='0.8')

plt.plot(x, y, 'xk') # raw data

plt.title('Conditional mean and variance')
plt.show()

#%%
# Evaluate log-likelihood for a range of lambda's
Q = 150
possible_lambdas = np.linspace(1, 30, Q)
possible_thetas = np.linspace(1, 30, Q)
loglikelihood = np.zeros((Q,Q))
for k in range(Q):
    lambda_k = possible_lambdas[k]
    for h in range(Q):
        theta_h = possible_thetas[h]
        K = kernel(x, x, lambda_k, theta_h) + sigma2*np.eye(N)
        (_, logdet) = np.linalg.slogdet(K)
        loglikelihood[k,h] = -0.5*N*np.log(2*np.pi) - 0.5*logdet - 0.5 * y.T.dot(np.linalg.pinv(K)).dot(y)

idx = np.argmax(loglikelihood)
(id_lambda, id_theta) = np.unravel_index(idx, loglikelihood.shape)
lambda_opt = possible_lambdas[id_lambda]
theta_opt = possible_thetas[id_theta]

#%%
plt.subplot(1, 2, 1)
plt.figure(1)
ax = plt.gca()
ax.grid(color='0.7', linestyle='-', linewidth=1)
plt.plot(possible_lambdas, loglikelihood[:, id_theta], '-k')
plt.plot(possible_lambdas[id_lambda], loglikelihood[id_lambda, id_theta], 'xk')
plt.title('Log-likelihood for lambda with theta = {}'.format(theta_opt));
plt.xlabel('lambda')
plt.ylabel('Log-likelihood')
plt.show()


#%%
#plt.subplot(1, 2, 1)
#plt.figure(1)
#
#ax = plt.gca()
#ax.grid(color='0.7', linestyle='-', linewidth=1)
#plt.plot(possible_thetas, loglikelihood[id_lambda, :], '-k')
#plt.plot(possible_thetas[id_theta], loglikelihood[id_lambda, id_theta], 'xk')
#plt.title('Log-likelihood for theta with lambda = {}'.format(lambda_opt));
#plt.xlabel('theta')
#plt.ylabel('Log-likelihood')
#plt.show()

