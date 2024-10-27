from __future__ import division
from nt_toolbox.general import *
from nt_toolbox.signal import *
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import skimage.io
import skimage.metrics
from numpy import linalg as LA
from nt_toolbox.perform_wavelet_transf import *
import pylops
# https://www.numerical-tours.com/matlab/denoisingwav_2_wavelet_2d/

def M(x, idx):
    return x[:, idx]

def MT(x, idx , nt):
    y = np.zeros((x.shape[0], nt))  # Modify the shape based on the use case
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

def psnr(img_1, img_2, data_range):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)





plt.close('all')


inputfile='data/matlab.mat'


data = scipy.io.loadmat(inputfile)
d = data['ds'] #matに保存した変数の呼び出し
d = d[:64,:64] 

n = 64
f0 = d
nx, nt = d.shape
dx, dt = 0.008, 0.004
x, t = np.arange(nx) * dx, np.arange(nt) * dt

sigma = 0.1
f1 = f0 + sigma*np.random.randn(n,n)

sampling_rate = 0.8
r = np.random.permutation(nt)[:math.ceil(nt*sampling_rate)]
initial = np.zeros((nx, nt,))
#print(initial.shape)
for c in range(math.ceil(nt*sampling_rate)):
    initial[:, r[c]] = f1[:, r[c]]

r.sort()
f1 = initial
idx = r

maxiter = 5
X = f1
gamma_1 = 0.9
gamma_2 = 1.1
beta = M(f1,idx)
epsilon = sigma * math.sqrt(X.size)
#options.ti = 1
#Jmin = 2
#T = 3.5*sigma

#Y = perform_wavelet_transf(X,Jmin,+1,ti=1)
slope = -pylops.utils.signalprocessing.slope_estimate(X.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
Y = Sop * X.ravel()
Y = Y.reshape(nx, nt)


nlevels_max = int(np.log2(64))
levels_size = np.flip(np.array([2 ** i for i in range(nlevels_max)]))
levels_cum = np.cumsum(levels_size)


print("epsilon:",epsilon)
for i in range(maxiter):  # In Python, loops are 0-indexed, so range(maxiter) is equivalent to 1:maxiter in MATLAB

    X_bef = X.copy()  # Ensure you copy the matrix rather than reference it
    
    # Step 1: Update X_tmp
    #X_tmp = X - gamma_1 * perform_wavelet_transf(Y, Jmin, -1,ti=1)
    slope = -pylops.utils.signalprocessing.slope_estimate(Y.T, dt, dx, smooth=6)[0]
    Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
    Sop_trans = Sop.transpose()
    X_sp = Sop_trans * Y.ravel()
    X_sp = X_sp.reshape(nx, nt)
    X_tmp = X - gamma_1 * X_sp


    # Step 2: Update X using MT, ProjL2ball, and M
    X = X_tmp + MT(ProjL2ball(M(X_tmp,idx), beta, epsilon) - M(X_tmp,idx), idx, nt)
    
    # Step 3: Update Y_tmp
    #Y_tmp = Y + gamma_2 * perform_wavelet_transf(2 * X - X_bef, Jmin, +1,ti=1)
    tmp = 2 * X - X_bef
    slope = -pylops.utils.signalprocessing.slope_estimate(tmp.T, dt, dx, smooth=6)[0]
    Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
    Y_sp = Sop * tmp.ravel()
    Y_sp = Y_sp.reshape(nx, nt)
    Y_tmp = Y + gamma_2 * Y_sp

    # Step 4: Update Y using Prox_l1norm
    Y = Y_tmp - gamma_2 * Prox_l1norm(Y_tmp / gamma_2, 1 / gamma_2)

    #if i % 10 == 0:
    print(i , " : " , psnr(d, X, 3))
    print(" fro : ", LA.norm(X - X_bef,"fro") / LA.norm(X,"fro")) #e-5以下で収束
    #print(" Ax : ", LA.norm(perform_wavelet_transf(X, Jmin, -1,ti=1),1))
    print(" Mx-B : ", LA.norm(X_bef - f1,2))


print("psnr : ",psnr(d, X, 3))


# Plot the results
plt.subplot(1, 3, 1)
plt.imshow(d, cmap='seismic', clim=(-0.1, 0.1))
#plt.imshow(d, cmap='seismic')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(f1, cmap='seismic', clim=(-0.1, 0.1))
#plt.imshow(f1, cmap='seismic')
plt.title('Lack')

plt.subplot(1, 3, 3)
plt.imshow(X, cmap='seismic', clim=(-0.1, 0.1))
#plt.imshow(X, cmap='seismic')
plt.title('Reconstructed')
plt.colorbar()
plt.show()



