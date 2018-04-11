import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import kernel_functions as kf
import pandas as pd


def kernel_wp_nonce_local(x, xp, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum

## Load data
## We subsample the data, which gives us N pairs of (x, y)
df = pd.read_excel('datasets/data_akbilgic.xlsx')

#data = loadmat('weather.mat')
N = len(df['USD BASED'].as_matrix())-1
x = np.arange(0, N, 5)
y = df['USD BASED'].as_matrix()[1:][x]
N = len(y)

## Standardize data to have zero mean and unit variance
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

## We want to predict values at x_* (denoted xs in the code)
M = 1000
xs = np.linspace(np.min(x), np.max(x), M)

## Data is assumed to have variance sigma^2 -- what happens when you change this number? (e.g. 0.1^2)
sigma2 = (1.0)**2

# Hyper parameter for kernel functions, to be optimized with max likelihood.
rho = 1

## Compute covariance (aka "kernel") matrices
kernel = kf.kernel_wp_nonce
K = kernel(x, x, rho) + sigma2*np.eye(N)
Ks = kernel(x, xs, rho)
Kss = kernel(xs, xs, rho)
 
## Compute conditional mean p(y_* | x, y, x_*)
Kinv = np.linalg.pinv(K)
mu = Ks.T.dot(Kinv).dot(y).astype(float)
Sigma = Kss - Ks.T.dot(Kinv).dot(Ks)

## Plot the mean prediction
plt.figure(1)
plt.plot(x, y, 'o-', markerfacecolor='k') # raw data
plt.plot(xs, mu) # mean prediction
plt.title('Mean prediction')
plt.show()

## Plot samples
plt.figure(2)
plt.plot(x, y, 'ko', markerfacecolor='k') # raw data
S = 50 # number of samples
samples = np.random.multivariate_normal(mu.reshape(-1), Sigma, S) # SxM
for s in range(S):
    plt.plot(xs, samples[s])
plt.title('Samples')
plt.show()

## Evaluate log-likelihood for a range of lambda's
Q = 1000
possible_rhos = np.linspace(0.1, 10, Q)
loglikelihood = np.zeros(Q)
for k in range(Q):
    rho_k = possible_rhos[k]
    K = kernel(x, x, rho_k) + sigma2*np.eye(N)
    (_, logdet) = np.linalg.slogdet(K)
    loglikelihood[k] = -0.5*N*np.log(2*np.pi) - 0.5*logdet - 0.5 * y.T.dot(np.linalg.pinv(K)).dot(y)

idx = np.argmax(loglikelihood)
rho_opt = possible_rhos[idx]
plt.figure(3)
plt.plot(possible_rhos, loglikelihood)
plt.plot(possible_rhos[idx], loglikelihood[idx], '*')
plt.title('Log-likelihood for \rho');
plt.xlabel('\rho')
plt.show()