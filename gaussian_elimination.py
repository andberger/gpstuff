import numpy as np
import kernel_functions as kf
import time
import matplotlib.pyplot as plt

def to_lower_echelon_form(A, m):
    for i in range(0, m-1):
        A[i] = A[i+1][i+1] * A[i] - A[i][i+1] * A[i+1]
    

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
    
def upper_row_echelon_form_to_identity(A, m):
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
            
def lower_row_echelon_form_to_identity(A, m):
    for i in range(m-1):
        factor = A[i, i] / A[m-1, i]
        A[i] = (1/factor) * A[i] 
    for i in reversed(range(1, m)):
        A[i] = A[i] - A[i-1]
    for i in range(m):
        A[i] = 1/A[i,i] * A[i]
    

        
    
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


#%%
   

#x = np.arange(20, 1020, 20)
x = np.array(list(sorted((50*np.random.random((50))))))

K = kf.kernel_wp_nonce(x, x, 1.0)
Kinv = get_inverse(K)
print(is_inverse(K, Kinv))


#%%

x_ls = np.arange(20, 1020, 20)
x_rnd = np.array(list(sorted((50*np.random.random((50))))))

K_ls = kf.kernel_wp_nonce(x_ls, x_ls, 1.0)
K_rnd = kf.kernel_wp_nonce(x_rnd, x_rnd, 1.0)

K_ls_inv = get_inverse(K_ls)
K_rnd_inv = get_inverse(K_rnd)

print(is_inverse(K_ls, K_ls_inv))
print(is_inverse(K_rnd, K_rnd_inv))


#%%
K_cached = None
K_cached_size = 0
def test(n=10, inverse_function=get_inverse):
    global K_cached_size
    global K_cached
    if(K_cached_size == n):
        K = K_cached
    else:
        x = np.arange(0, n*10, 10)
        x = (x-np.mean(x)) / np.std(x)
        #print(x)
        K = kf.kernel_wp_nonce(x, x, 1.0)
        K_cached_size = n
        K_cached = K
        
    runs = 0 
    start_time = time.time()
    while True:
        runs += 1
        Kinv = inverse_function(K)
        if time.time() - start_time > 2:
            break
    
    time_per_run = (time.time()-start_time)/runs
    return time_per_run


ns2= [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
times1 = [0.00024018999937059328,
 0.001172351389948676,
 0.0028016417176573425,
 0.008094185783017066,
 0.020362292877351394,
 0.06063500317660245,
 0.41081376870473224,
 1.360371470451355,
 5.861602067947388]

times2 = [0.00018669940724928648,
 0.0008185730877585694,
 0.0024366500752271312,
 0.020629986045286826,
 0.09150949391451749,
 0.5706148743629456,
 13.515231847763062,
 168.7982976436615]
times = []


ns = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 9000]

plt.figure()

our_time = []
pinv_time = []
for n in ns:
    print("Starting test for n={}".format(n))
    t = test(n=n)
    our_time.append(t)
    t2 = test(n=n, inverse_function=np.linalg.pinv)
    pinv_time.append(t2)
    
    plt.plot(ns[:len(our_time)], our_time, label='get_inverse')
    plt.plot(ns[:len(pinv_time)], pinv_time, label='pinv')
    plt.legend()
    plt.yscale('log', basey=2)
    plt.xscale('log', basex=2)
    plt.show()

    print("........ test for n={} took {} per run".format(n, t))
    

#%%



#%%
import numpy as np
import matplotlib.pyplot as plt

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def once_inversion(A):
    ''' A is a once integrated wiener process covariance matrix (linspaced) '''
    m = len(A)
    for i in range(m-2):
        alpha1 = A[i, i+1] - A[i, i]
        alpha2 = A[i+1, i+2] - A[i+1, i+1]
        A[i] = A[i] - (alpha1/alpha2) * A[i+1]
    for i in range(m-2):
        over = A[i, i+1]
        under = A[i+1, i+2]
        A[i] = A[i] - (over/under) * A[i+1]
        
x = np.arange(20, 20*11, 20)
#x = np.array(list(sorted((50*np.random.random((50))))))
x = (x - np.mean(x)) / np.std(x)
K = kf.kernel_wp_once(x, x, 1.0)
#print(K)

plt.imshow(K);
plt.colorbar()
plt.show()

once_inversion(K)
print(K)

plt.imshow(K);
plt.colorbar()
plt.show()

for i in range(10):
    plt.plot(K[i], label=i)
plt.legend()
plt.show()


#%%

def swap_rows(i, j, A):
    A[[i, j]] = A[[j, i]]
    
def explore(A, step):
    print("Step {}".format(step))
    B = A.copy()
    B[B<-10] = -10
    B[B>10] = 10
    plt.imshow(B)
    plt.colorbar()
    plt.show()

def once_inversion_success(A):
    '''A is a once integrated wiener process covariance matrix (linspaced) and 0<a<b<c... where a=a, b=2a, c=3a... '''
    m = len(A)
    
    for i in range(m-1):
        A[i] = A[i] - A[i+1]
    
    explore(A, 1)

    for i in range(m-1):
        A[i] = A[i] - A[i+1]
        
    explore(A, 2)
    
    A[m-2] = A[m-2] - (A[m-2, 0] / A[m-1, 0]) * A[m-1] 
    A[m-1] = A[m-1] - (A[m-1, m-1] / A[m-2, m-1]) * A[m-2]
    
    explore(A, 3)
    #A = A*1000
    
    for i in range(1, m-1):
        print(A[i-1, i])
        A[m-2] = A[m-2] - (A[m-2, i] / A[i-1, i]) * A[i-1]
        A[m-1] = A[m-1] - (A[m-1, i] / A[i-1, i]) * A[i-1]
        
    explore(A, 4)
    
    for i in reversed(range(0, m-1)):
        swap_rows(i, i+1, A)
    explore(A, 5)
    
    for i in range(m):
        A[i] = (1/A[i,i]) * A[i]
        
    explore(A, 6)
    
    for i in range(m-1):
        A[i] = A[i] - A[i+1]
        
    explore(A, 7)
    
    for i in range(m-1):
        A[i] = A[i] - A[i+1]
        
    explore(A, 8)    
    
    for i in reversed(range(1, m-2)):
        #print(A[0, i+2], A[i, i+2], (A[0, i+2]/A[i, i+2]))
        A[0] = A[0] - (A[0, i+2]/A[i, i+2]) * A[i]
        
    explore(A, 9)
    
        
    
        
      
def once_inversion_success_complete(A):
    '''A is a once integrated wiener process covariance matrix (linspaced) and 0<a<b<c... where a=a, b=2a, c=3a... '''
    m = len(A)
#    
#    for i in range(m-1):
#        A[i] = A[i] - A[i+1]
#
#    for i in range(m-1):
#        A[i] = A[i] - A[i+1]
#
#    A[m-2] = A[m-2] - (A[m-2, 0] / A[m-1, 0]) * A[m-1] 
#    A[m-1] = A[m-1] - (A[m-1, m-1] / A[m-2, m-1]) * A[m-2]
#    
#    for i in range(1, m-1):
#        A[m-2] = A[m-2] - (A[m-2, i] / A[i-1, i]) * A[i-1]
#        A[m-1] = A[m-1] - (A[m-1, i] / A[i-1, i]) * A[i-1]
#        
#    for i in reversed(range(0, m-1)):
#        swap_rows(i, i+1, A)
#    
#    for i in range(m):
#        A[i] = (1/A[i,i]) * A[i]
#    
#    for i in range(m-1):
#        A[i] = A[i] - A[i+1]
#    
#    for i in range(m-1):
#        A[i] = A[i] - A[i+1]
#        
#
#    for i in reversed(range(1, m-1)):
#        print(A[i+1, i+1])
#        A[i] = A[i] - (A[i, i+1]/A[i+1, i+1]) * A[i+1]
#        A[i-1] = A[i-1] - (A[i-1, i+1]/A[i+1, i+1]) * A[i+1]
#
#    for i in reversed(range(1, m)):
#        A[0] = A[0] - (A[0, i]/A[i, i]) * A[i]
#            
        

#K = np.array([[2.0,5.0,8.0,11.0], [5.0,4.0,7.0,10.0], [8.0,7.0,6.0,9.0], [11.0,10.0,9.0,8.0]])

x = np.arange(20, 20*11, 20)
x = x 
x = (x - np.mean(x)) / np.std(x)
K = kf.kernel_wp_once(x, x, 1.0)

once_inversion_success_complete(K)

print(np.round(K, decimals=3))
K[K<-2] = -2
K[K>2] = 2
plt.imshow(K)
plt.colorbar()
plt.show()


#%%
a = np.power(6,1/3)
x = np.arange(a, a*7, a)
#x = (x - np.mean(x)) / np.std(x)
K = kf.kernel_wp_once(x, x, 1.0)     

print(np.round(K, decimals=2))















