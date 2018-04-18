import numpy as np

def gaussian_elimination(A, m, n=0):
    if(n == 0):
        n = m
    eps = 0.00000000000000001;
    h = 0
    k = 0
    while h < m and k < n:
        #print(A, h, k)
        i_max = h
        v_max = 0
        for i in range(h, m):
            v = abs(A[i, k])
            if v > v_max:
                i_max = i
                v_max = v
        
        if abs(A[i_max, k]) <= eps:
            k = k+1
        else:
            swap_rows(h, i_max, A)
            #print("After swap\n", A)
            for i in range(h+1, m):
                f = A[i, k] / A[h, k]
                #print("f = {}/{} = {}".format(A[i, k], A[h, k], f))
                A[i, k] = 0.0
                A[i, (k+1):] = A[i, (k+1):] - A[h, (k+1):]*f
                #print("After subtraction\n", A)
            h = h+1
            k = k+1
            
def swap_rows(i, j, A):
    A[[i, j]] = A[[j, i]]
    
def row_echelon_form_to_identity(A, m):
    n = m-1
    for i in range(0, m):
        if n-i == n:
            A[n-i] = A[n-i] * 1/A[n-i][n-i]
        elif n-i == 0:
            for j in range(0, m):
                if n-j != 0:
                    A[n-i] = A[n-i] * 1/A[n-i][n-i + n-j] - A[n-j]
                else:
                    A[n-i] = A[n-i] * 1/A[n-i][n-i + n-j]
        else:
            for k in range(0, i):
                A[n-i] = A[n-i] - A[n-i + k+1] * A[n-i][n-i + k+1]
            A[n-i] = A[n-i] * 1/A[n-i][n-i]
        
    
def get_inverse(A):
    """
    A assumed square
    """
    A_I = np.hstack((A, np.eye(len(A))))
    gaussian_elimination(A_I, len(A))
    row_echelon_form_to_identity(A_I, len(A))
    return np.array(A_I[:, len(A):len(A)*2])

    
def is_inverse(A, A_inv):
    eps = 0.00001
    diff =  np.matmul(A, A_inv) - np.eye(len(A))
    return np.sum(diff**2) < eps





