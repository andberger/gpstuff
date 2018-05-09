import numpy as np
import kernel_functions as kf

def lower_row_echelon_form_to_identity(A, m):
    for i in range(m-1):
        factor = A[i, i] / A[m-1, i]
        A[i] = (1/factor) * A[i] 
    for i in reversed(range(1, m)):
        A[i] = A[i] - A[i-1]
    for i in range(m):
        A[i] = 1/A[i,i] * A[i]
        
def to_lower_echelon_form(A, m):
    for i in range(0, m-1):
        A[i] = A[i+1][i+1] * A[i] - A[i][i+1] * A[i+1]
        
def get_inverse(A):
    """
    A assumed square
    """
    A_I = np.hstack((A, np.eye(len(A))))
    to_lower_echelon_form(A_I, len(A))
    lower_row_echelon_form_to_identity(A_I, len(A))
    return np.array(A_I[:, len(A):len(A)*2])

    
def is_inverse(A, A_inv):
    eps = 0.00001
    diff =  np.matmul(A, A_inv) - np.eye(len(A))
    return np.sum(diff**2) < eps




x_ls = np.arange(20, 1020, 20)
x_rnd = np.array(list(sorted((50*np.random.random((50))))))

K_ls = kf.kernel_wp_nonce(x_ls, x_ls, 1.0)
K_rnd = kf.kernel_wp_nonce(x_rnd, x_rnd, 1.0)

K_ls_inv = get_inverse(K_ls)
K_rnd_inv = get_inverse(K_rnd)

print(is_inverse(K_ls, K_ls_inv))
print(is_inverse(K_rnd, K_rnd_inv))