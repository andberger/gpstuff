import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from itertools import chain
#import kernel_functions as kf


"""
nonce struktur:
    array([[-1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53],
           [-1.53, -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 ],
           [-1.53, -1.1 , -0.66, -0.66, -0.66, -0.66, -0.66, -0.66],
           [-1.53, -1.1 , -0.66, -0.22, -0.22, -0.22, -0.22, -0.22],
           [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.21,  0.21,  0.21],
           [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.65,  0.65,  0.65],
           [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.65,  1.09,  1.09],
           [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.65,  1.09,  1.52]])
    
 
x = [a b c d e f g]

K = 
[a a a a]
[a b b b]
[a b c c] 
[a b c d]

y = [r s t u]
K * y = 
[
a*(r+s+t+u)
a*r + b*(s+t+u)
a*r+b*s + c*(t+u)
a*r+b*s+c*t + d*(u)   
]
=



struktur = array([[-0.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53], ...
       						[-1.53, -0.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 , -1.1 ], ...
                  [-1.53, -1.1 ,  0.34, -0.66, -0.66, -0.66, -0.66, -0.66], ...
                  [-1.53, -1.1 , -0.66,  0.78, -0.22, -0.22, -0.22, -0.22], ...
                  [-1.53, -1.1 , -0.66, -0.22,  1.21,  0.21,  0.21,  0.21], ...
                  [-1.53, -1.1 , -0.66, -0.22,  0.21,  1.65,  0.65,  0.65], ...
                  [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.65,  2.09,  1.09], ...
                  [-1.53, -1.1 , -0.66, -0.22,  0.21,  0.65,  1.09,  2.52]])

struktur - I =
a a a a a 
a b b b b
a b c c c
a b c d d
a b c d e

[0.0 0.1 0.1 0.1 0.1 ....
 0.1 1.1 1.2 1.2 1.2 ....
 0.1 1.2 2.2 2.3     .... ]


# thessu
[ K       * y
 a b b b b  u
 b c d d d  v
 b d e f f  x
 b d f g h  y
 b d f h i  z
]


A
 . b b b b
 b . d d d
 b d . f f
 b d f . h
 b d f h . 

 B
 a . . . .
 . c . . .
 . . e . .
 . . . g .
 . . . . i 

A*y + B*y = K*y

y = [u v x y z]

a0 = 0
a1 = a0 + b*u
a2 = a1 + d*v
a3 = a2 + f*x


b0 = b1 + v  = v+y+x
b1 = b2 + x  = y+x
b2 = y       = y
b3 = 0

A*s + B*s = K*s =
[ 0*u + b*v + b*x + b*y,     = 0  + b*b0  + a*u
  b*u + 0*v + d*x + d*y,     = a1 + d*b1  + c*v
  b*u + d*v + 0*x + f*y,     = a2 + f*b2  + e*x
  b*u + d*v + f*x + 0*y ]    = a3 + 0     + g*y

K*s = a + [b d f 0]*b + [a c e g]*s
O(K*s) = O(a) + O([b d f 0]*b) + O([a c e g]*s)
  
minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))


 A
 . b b b b
 b . d d d
 b d . f f
 b d f . h
 b d f h . 

 B
 a . . . .
 . c . . .
 . . e . .
 . . . g .
 . . . . i 
 
 
a = min(x[0], xp[0])

######
x =  [3 1 3]
xp = [1 0 2]

K =
[1 0 2
 1 0 1
 1 0 2]

min(x[0], xp[0]) = 1
min(x[1], xp[0]) = 2
min(x[1], xp[1]) = 3

[1 2
 2 3]
######

              vigur1
        . . . . . . . . .
        .
vigur2  .
        .
        .
        .
        .
       
####
"""


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

def gaussian_elimination(A):
    h = 1
    k = 1
    m = len(A)
    n = len(A)
    while h <= m and k <= n:
        i_max = max(list(chain.from_iterable((i, abs(A[i,k-1])) for i in range(1, m))))
        if A[i_max, k-1] == 0:
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
x = x[0:8]
y = y[0:8]
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
Kinv = np.linalg.pinv(K)
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

