import numpy as np
import matplotlib.pyplot as plt
import kernel_functions as kf
import gaussian_elimination as ge

x = np.arange(0, 1000, 20)
x = (x - np.mean(x)) / np.std(x)
N = len(x)
sigma2 = (1.0)**2
rho = 1.0


nonce_4x4_inv = ge.get_inverse(kf.kernel_wp_nonce(x[:4], x[:4], rho) + sigma2*np.eye(4))
nonce_10x10_inv = ge.get_inverse(kf.kernel_wp_nonce(x[:10], x[:10], rho) + sigma2*np.eye(10))
nonce_50x50_inv = ge.get_inverse(kf.kernel_wp_nonce(x, x, rho) + sigma2*np.eye(N))

plt.matshow(nonce_4x4_inv)
plt.show()
plt.plot(nonce_4x4_inv[0,:])
plt.show()

plt.matshow(nonce_10x10_inv)
plt.show()
plt.plot(nonce_10x10_inv[0,:])
plt.show()

plt.matshow(nonce_50x50_inv)
plt.show()
plt.plot(nonce_50x50_inv[0,:])
plt.show()

