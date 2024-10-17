from __future__ import division
from nt_toolbox.general import *
from nt_toolbox.signal import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pylops
import scipy
import skimage.io
import skimage.metrics
from numpy import linalg as LA


plt.close('all')


inputfile='data/matlab.mat'


data = scipy.io.loadmat(inputfile)
d = data['ds'] #matに保存した変数の呼び出し
d = d[:64,:64]


rng = np.random.default_rng()
v = rng.random(4096)
w = rng.random(4096)

nx, nt = d.shape
dx, dt = 8, 0.004 #初期値
x, t = np.arange(nx) * dx, np.arange(nt) * dt

print("v: ",v)
print("w: ",w)
print("transpose()")

v = v.reshape(nx,nt)
w = w.reshape(nx,nt)
slope = -pylops.utils.signalprocessing.slope_estimate(v.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
slope = -pylops.utils.signalprocessing.slope_estimate(w.T, dt, dx, smooth=6)[0]
Sop_trans = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).transpose()
Sop_adj = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).adjoint()
Sop_H = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).H

v = v.ravel()
w = w.ravel()

#print(Sop.shape)
#print(Sop_trans.shape)
print("〈Av,w〉",np.dot(Sop * v.ravel() , w))
print("〈v,ATw〉",np.dot(v, Sop_trans * w))
print("〈v,ATw〉",np.dot(v, Sop_adj * w))
print("〈v,ATw〉",np.dot(v, Sop_H * w))
#print(np.dot(Sop * v.ravel() , w) == np.dot(v, Sop_trans * w))

print("smooth = 1")

v = v.reshape(nx,nt)
w = w.reshape(nx,nt)
slope = -pylops.utils.signalprocessing.slope_estimate(v.T, dt, dx, smooth=1)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
slope = -pylops.utils.signalprocessing.slope_estimate(w.T, dt, dx, smooth=1)[0]
Sop_trans = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).transpose()
Sop_adj = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).adjoint()
Sop_H = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).H

v = v.ravel()
w = w.ravel()

#print(Sop.shape)
#print(Sop_trans.shape)
print("〈Av,w〉",np.dot(Sop * v.ravel() , w))
print("〈v,ATw〉",np.dot(v, Sop_trans * w))
print("〈v,ATw〉",np.dot(v, Sop_adj * w))
print("〈v,ATw〉",np.dot(v, Sop_H * w))

"""
w = np.zeros(4096)
w[0] = 1
w = w.reshape(nx,nt)

slope = - pylops.utils.signalprocessing.slope_estimate(w.T, dt, dx, smooth=10)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
w_tmp = Sop * w.ravel()
w_tmp = w_tmp.reshape(nx,nt)

slope = - pylops.utils.signalprocessing.slope_estimate(w_tmp.T, dt, dx, smooth=10)[0]
Sop_H = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt)).H
w_SOP = Sop_H * w_tmp.ravel()
w_SOP = w_SOP.reshape(nx,nt)

print(w_SOP)

plt.imshow(w_SOP)
plt.colorbar()
plt.show()
"""


print("inverse()")

v = v.reshape(nx,nt)
w = w.reshape(nx,nt)
slope = -pylops.utils.signalprocessing.slope_estimate(v.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
seis = Sop * v.ravel() #〈Av,w〉

slope = -pylops.utils.signalprocessing.slope_estimate(w.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
drec = Sop.inverse(w.ravel())#〈v,ATw〉
#seis = seis.reshape(nx,nt)
#drec = drec.reshape(nx,nt)

#print(seis.shape)
#print(drec.shape)

print("〈Av,w〉",np.dot(seis , w.ravel()))
print("〈v,ATw〉",np.dot(v.ravel(), drec))
#print(np.dot(seis * v , w) == np.dot(v, drec * w))
#print(type(Sop))

print("inv")
slope = -pylops.utils.signalprocessing.slope_estimate(d.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt),inv=False)
Sop_inv = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt),inv=True)


print("〈Av,w〉",np.dot(Sop * v.ravel() , w))
print("〈v,ATw〉",np.dot(v, Sop_inv * w))
print(np.dot(Sop * v.ravel() , w) == np.dot(v, Sop_inv * w))


# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# axs[0].imshow(slope)
# axs[1].imshow(Sop.slopes)
# plt.show()

print("H")
slope = -pylops.utils.signalprocessing.slope_estimate(d.T, dt, dx, smooth=6)[0]
Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt),inv=False)
Sop_h = Sop.H


print("〈Av,w〉",np.dot(Sop * v.ravel() , w))
print("〈v,ATw〉",np.dot(v, Sop_h * w))
print(np.dot(Sop * v.ravel() , w) == np.dot(v, Sop_h * w))








print("Check if Orthogonal")

v1 = np.zeros(4096)
v1[0] = 1
v2 = np.zeros(4096)
v2[1] = 1
v3 = np.zeros(4096)
v3[2] = 1
v4 = np.zeros(4096)
v4[3] = 1

print("v1 : ",v1)
print("v2 : ",v2)
print("v3 : ",v3)
print("v4 : ",v4)

Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))
w1 = Sop * v1.ravel() 
w2 = Sop * v2.ravel() 
w3 = Sop * v3.ravel() 
w4 = Sop * v4.ravel() 

print("w1 : ",w1)
print("w2 : ",w2)
print("w3 : ",w3)
print("w4 : ",w4)
print("w1・w1",np.dot(w1,w1))
print("w2・w2",np.dot(w2,w2))
print("w3・w3",np.dot(w3,w3))
print("w4・w4",np.dot(w4,w4))
print("w1・w2 : ",np.dot(w1,w2))
print("w2・w3 : ",np.dot(w2,w3))
print("w1・w3 : ",np.dot(w1,w3))
