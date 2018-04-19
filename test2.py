# -*- coding: utf-8 -*-
import numpy as np
from random import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def kernel_wp_nonce_local(x, xp, y, sigma):
    minimum = cdist(x.reshape((-1, 1)), xp.reshape((-1, 1)), lambda u, v: np.fmin(u,v))
    return (sigma**2) * minimum


d = np.sort(np.random.rand((5)))
e = np.sort(np.random.rand((7)))
res = np.round(kernel_wp_nonce_local(d, e, 0, 1), decimals=2)

print(np.round(d, decimals=2))
print(np.round(e, decimals=2))
print(res)


plt.matshow(res)

plt.show()



res = np.round(kernel_wp_nonce_local(d, d, 0, 1), decimals=2)
print(res)
plt.matshow(res)

plt.show()


print("-------------")

def invert(m):
    A = np.zeros(m.shape)
    for i in range(m.shape[0]):
        print(i)
        if i==0:
            a = m[0, 0]
            b = m[1, 1]
            bp = 1/(b-a)
            A[0, 0] = 1/a + bp
            A[0, 1] = -bp
            print(1/a + bp)
            print(-bp)
        elif i==m.shape[0]-1:
            a = m[i-1, i-1]
            b = m[i, i]
            bp = 1/(b-a)
            A[i, i-1] = -bp
            A[i,  i] = bp
            print(-bp)
            print(bp)
        else:
            a = m[i-1, i-1]
            b = m[i, i]
            c = m[i+1, i+1]
            bp = 1/(b-a)
            cp = 1/(c-b)
            A[i, i-1] = -bp
            A[i,  i] = bp+cp
            A[i, i+1] = -cp
            print(-bp)
            print( bp+cp)
            print(-cp)
            
    return A


a = random()
b = random()
c = random()
d = random()
e = random()

m = np.array([[a, a, a, a, a], 
              [a, b, b, b, b], 
              [a, b, c, c, c], 
              [a, b, c, d, d], 
              [a, b, c, d, e]])
print(m)
print("A")
print(invert(m))
print(np.round(np.matmul(m, invert(m)), decimals=2))

print("-------------")

def invert_and_dot(m, y):
    if(len(m) is not len(y)):
        print("Error, vectors not of the same length")
    
    res = []
    for i in range(len(m)):
        val = 0
        if i==0:
            print("first", i)
            bp = 1/(m[i])
            cp = 1/(m[i+1]-m[i])
            print(bp, cp)
            val = (bp+cp)*y[i] - cp*y[i+1]
        elif i==len(m)-1:
            print("last", i)
            bp = 1/(m[i]-m[i-1])
            val = -bp*y[i-1] + (bp)*y[i]
        else:
            print("middle", i)
            bp = 1/(m[i]-m[i-1])
            cp = 1/(m[i+1]-m[i])
            val = -bp*y[i-1] + (bp+cp)*y[i] - cp*y[i+1]
        res.append(val)
    return res
    
m = np.random.rand((5))
y = np.random.rand((5))
#invert_and_dot(m, y)


K = kernel_wp_nonce_local(m, m, 0, 1)
res1 = K.dot(y)
res2 = invert_and_dot(m, y)

print(m)
print(y)
print(np.round(res1, decimals=3))
print(np.round(res2, decimals=3))