# -*- coding: utf-8 -*-

import numpy as np
from random import random

print("3x3 ................" )
a = random()
b = random()
c = random()

A = np.array([[a, a, a], 
              [a, b, b], 
              [a, b, c]])

ap = 1/a
bp = 1/(b-a)
cp = 1/(c-b)

B = np.array([[ap+bp  , -bp    , 0   ],
              [-bp    , bp+cp  , -cp ],
              [0      , -cp    , cp  ]])

C = np.matmul(A, B)

print(np.round(A, decimals=2))
print(np.round(B, decimals=2))
print(np.round(C, decimals=10))




print("4x4 ................" )
a = random()
b = random()
c = random()
d = random()

A = np.array([[a, a, a, a], 
              [a, b, b, b], 
              [a, b, c, c], 
              [a, b, c, d]])

ap = 1/a
bp = 1/(b-a)
cp = 1/(c-b)
dp = 1/(d-c)

B = np.array([[ap+bp  , -bp    , 0     ,   0  ],
              [-bp    , bp+cp  , -cp   ,   0  ],
              [0      , -cp    , cp+dp ,  -dp ],
              [0      , 0      , -dp   ,   dp ]])

C = np.matmul(A, B)

print(np.round(A, decimals=2))
print(np.round(B, decimals=2))
print(np.round(C, decimals=10))







print("5x5 ................" )
a = random()
b = random()
c = random()
d = random()
e = random()

A = np.array([[a, a, a, a, a], 
              [a, b, b, b, b], 
              [a, b, c, c, c], 
              [a, b, c, d, d], 
              [a, b, c, d, e]])

ap = 1/a
bp = 1/(b-a)
cp = 1/(c-b)
dp = 1/(d-c)
ep = 1/(e-d)

B = np.array([[ap+bp  , -bp    , 0     ,   0   , 0   ],
              [-bp    , bp+cp  , -cp   ,   0   , 0   ],
              [0      , -cp    , cp+dp ,  -dp  , 0   ],
              [0      , 0      , -dp   , dp+ep , -ep ],
              [0      , 0      , 0     ,   -ep , ep  ]])

C = np.matmul(A, B)

print(np.round(A, decimals=2))
print(np.round(B, decimals=2))
print(np.round(C, decimals=10))

