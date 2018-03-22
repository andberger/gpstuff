import numpy as np
from scipy.spatial.distance import cdist


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