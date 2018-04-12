import numpy as np

def gaussian_elimination(A, m, n):
    eps = 0.00000000000000001;
    h = 0
    k = 0
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
                A[i, (k+1):] = A[i, (k+1):] - A[h, (k+1):]*f
                print("After subtraction\n", A)
            h = h+1
            k = k+1
            
def swap_rows(i, j, A):
    A[[i, j]] = A[[j, i]]
    
def row_echelon_form_to_identity(A, m):
    n = m-1
    for i in range(0,m):
        if n-i == n:
            A[n-i] = A[n-i] * 1/A[n-i][n-i]
        elif n-i == 0:
            for j in range(0, m):
                if n-j != 0:
                    A[n-i] = A[n-i] * 1/A[n-i][n-i + n-j] - A[n-j]
                else:
                    A[n-i] = A[n-i] * 1/A[n-i][n-i + n-j]
        else:
            for k in range(1, n):
                A[n-i] = A[n-i] - A[n-i + k] * A[n-i][n-i + k]
                A[n-i] = A[n-i] * 1/A[n-i][n-i]
        
    
    


A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
A_I = np.hstack((A,np.eye(len(A))))
print(A)
print(A_I)
B = np.array(A)
B_I = np.array(A_I)
gaussian_elimination(B, 3, 3)
gaussian_elimination(B_I, 3, 3)
print(B)
print(B_I)