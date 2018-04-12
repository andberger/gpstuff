import numpy as np

def gaussian_elimination(A):
    eps = 0.00000000000000001;
    h = 0
    k = 0
    m = len(A[0])
    n = len(A)
    while h < m and k < n:
        print(A, h, k)
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
            print("After swap\n", A)
            for i in range(h+1, m):
                f = A[i, k] / A[h, k]
                print("f = {}/{} = {}".format(A[i, k], A[h, k], f))
                A[i, k] = 0.0
                A[i, (k+1):n] = A[i, (k+1):n] - A[h, (k+1):n]*f
                print("After subtraction\n", A)
            h = h+1
            k = k+1
            
def swap_rows(i, j, A):
    A[[i, j]] = A[[j, i]]
    


A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
print(A)
gaussian_elimination(A)
print(A)