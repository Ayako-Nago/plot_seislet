"""
Seislet transform
=================
This example shows how to use the :py:class:`pylops.signalprocessing.Seislet`
operator. This operator the forward, adjoint and inverse Seislet transform
that is a modification of the well-know Wavelet transform where local slopes
are used in the prediction and update steps to further improve the prediction
of a trace from its previous (or subsequent) one and reduce the amount of
information passed to the subsequent scale. While this transform was initially
developed in the context of processing and compression of seismic data, it is
also suitable to any other oscillatory dataset such as GPR or Acoustic
recordings.

"""
from __future__ import division
from nt_toolbox.general import *
from nt_toolbox.signal import *
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

def M(x, idx):
    return x[:, idx]

def MT(x, idx):
    # Assume y is a pre-allocated array of zeros or a desired shape
    y = np.zeros((x.shape[0], max(idx) + 1))  # Modify the shape based on the use case
    y[:, idx] = x
    return y


def ProjL2ball(u, f, epsilon):
    temp = u - f
    radius = np.linalg.norm(temp.ravel(), 2)  # Flatten the array and compute the L2 norm
    if radius > epsilon:
        u = f + (epsilon / radius) * temp
    return u

def Prox_l1norm(A, gamma):
    return np.sign(A) * np.maximum(np.abs(A) - gamma, 0)





plt.close('all')


inputfile='data/matlab.mat'

#d = np.load(inputfile)
data = scipy.io.loadmat(inputfile)
d = data['ds'] #matに保存した変数の呼び出し
d = d[:64,:64] 

n = 64
f0 = d


sigma = 0.1
f1 = f0 + sigma*np.random.randn(n,n)

sampling_rate = 0.8
r = np.random.permutation(n)[:math.ceil(n*sampling_rate)]
initial = np.zeros((n,n,))

for c in range(math.ceil(n*sampling_rate)):
    initial[:, r[c]] = f1[:, r[c]]

r.sort()
f1 = initial
idx = r

maxiter = 20000
X = f1
gamma_1 = 0.9
gamma_2 = 1.1
beta = f1
epsilon = sigma * math.sqrt(X.size)
#options.ti = 1
Jmin = 2
T = 3.5*sigma

#hについて調べる必要あり
h = [0, .482962913145, .836516303738, .224143868042, -.129409522551]
h = h/np.linalg.norm(h)

Y = perform_wavortho_transf(X,Jmin,+1,h)

for i in range(maxiter):  # In Python, loops are 0-indexed, so range(maxiter) is equivalent to 1:maxiter in MATLAB
    X_bef = X.copy()  # Ensure you copy the matrix rather than reference it
    
    # Step 1: Update X_tmp
    X_tmp = X - gamma_1 * perform_wavortho_transf(Y, Jmin, -1, h)
    
    # Step 2: Update X using MT, ProjL2ball, and M
    X = X_tmp + MT(ProjL2ball(M(X_tmp,idx), beta, epsilon) - M(X_tmp,idx), idx)
    
    # Step 3: Update Y_tmp
    Y_tmp = Y + gamma_2 * perform_wavortho_transf(2 * X - X_bef, Jmin, +1, h)
    
    # Step 4: Update Y using Prox_l1norm
    Y = Y_tmp - gamma_2 * Prox_l1norm(Y_tmp / gamma_2, 1 / gamma_2)





# Plot the results
plt.subplot(1, 3, 1)
plt.imshow(d, cmap='seismic', clim=(-0.1, 0.1))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(f1, cmap='seismic', clim=(-0.1, 0.1))
plt.title('Lack')

plt.subplot(1, 3, 3)
plt.imshow(d, cmap='seismic', clim=(-0.1, 0.1))
plt.title('Reconstructed')
plt.colorbar()
plt.show()



