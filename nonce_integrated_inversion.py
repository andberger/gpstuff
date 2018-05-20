import numpy as np
import kernel_functions as kf
import matplotlib.pyplot as plt

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

def get_inverse_and_plot(A):
    """
    A assumed square
    """
    plot_matrix(A, "A a square matrix")
    A_I = np.hstack((A, np.eye(len(A))))
    plot_matrix(A_I, "Augmented with the identity matrix")
    to_lower_echelon_form(A_I, len(A))
    plot_matrix(A_I, "to lower echelon form")
    lower_row_echelon_form_to_identity(A_I, len(A))
    plot_matrix(A_I, "to identity")
    plot_matrix(np.array(A_I[:, len(A):len(A)*2]), "The inverse of A")
    return np.array(A_I[:, len(A):len(A)*2])

    
def is_inverse(A, A_inv):
    eps = 0.00001
    diff =  np.matmul(A, A_inv) - np.eye(len(A))
    return np.sum(diff**2) < eps


def plot_matrix(A, info_text):
    print("Info: {}".format(info_text))
    B = A.copy()
    B[B<-n] = -n
    B[B>n] = n
    plt.imshow(B)
    plt.colorbar(boundaries=list(range(-n,n)))
    plt.show()

n=20
x_ls = np.arange(20, 1020, 20)
x_rnd = np.array(list(sorted(n*np.random.random((n)))))

K_ls = kf.kernel_wp_nonce(x_ls, x_ls, 1.0)
K_rnd = kf.kernel_wp_nonce(x_rnd, x_rnd, 1.0)

K_ls_inv = get_inverse(K_ls)
K_rnd_inv = get_inverse_and_plot(K_rnd)

print(is_inverse(K_ls, K_ls_inv))
print(is_inverse(K_rnd, K_rnd_inv))