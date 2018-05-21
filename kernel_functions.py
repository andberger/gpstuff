import numpy as np
from scipy.spatial.distance import cdist


def kernel_wp_nonce(x, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum

def kernel_wp_once(x, y, sigma):
    dist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'euclidean')
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    term1 = np.power(minimum, 3) / 3
    term2 = np.power(minimum, 2) / 2
    
    return (sigma**2) * (term1 + dist*term2)

def kernel_wp_twice(x, y, sigma):
    dist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'euclidean')
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    term1 = np.power(minimum, 5) / 20
    term2 = np.power(minimum, 3)
    term3 = np.power(minimum, 4) / 2
    
    return (sigma**2) * (term1 + (dist/12) * ((x + y) * term2 - term3))

def kernel_wp_thrice(x, y, sigma):
    dist = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), 'euclidean')
    minimum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    maximum = cdist(x.reshape((-1, 1)), y.reshape((-1, 1)), lambda u, v: np.fmax(u,v))
    term1 = np.power(minimum, 7) / 252
    term2 = np.power(minimum, 4)
    term3 = np.power(minimum, 2)
    
    return (sigma**2) * ((term1 + ((dist*term2)/720)) * (5*np.power(maximum, 2) + 2*x*y + 3*term3))
