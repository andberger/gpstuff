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
        
        if A[i_max, k] <= eps:
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
    

#3x3
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
#4x4
A = np.array([[1.0, 2.0, 3.0, 6.0], [4.0, 5.0, 6.0, 13.0], [7.0, 8.0, 10.0, 17.0], 
              [9.0, 10.0, 11.0, 19.0]])
A_I = np.hstack((A,np.eye(len(A))))
print(A)
print(A_I)
B = np.array(A)
B_I = np.array(A_I)
B_I_rc_count = np.shape(B_I)[0]
gaussian_elimination(B, 3, 3)
gaussian_elimination(B_I, B_I_rc_count, B_I_rc_count)
row_echelon_form_to_identity(B_I, B_I_rc_count)
print(B)
print(B_I)
B_I_final = B_I[:, B_I_rc_count:B_I_rc_count*2]
print("------------------------------")
print(np.round(B_I_final, decimals=2))
print(np.round(np.matmul(A, B_I_final), decimals=2))