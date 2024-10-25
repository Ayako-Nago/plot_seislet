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



rng = np.random.default_rng()
v = rng.random(4096)
y = rng.random(4096)

nx, nt = 64,64
dx, dt = 8, 0.004 #初期値
x, t = np.arange(nx) * dx, np.arange(nt) * dt


v = v.reshape(nx,nt)
y = y.reshape(nx,nt)


sum = v + y



slope_x = -pylops.utils.signalprocessing.slope_estimate(v.T, dt, dx, smooth=6)[0]
Sop_x = pylops.signalprocessing.Seislet(slope_x.T, sampling=(dx, dt))

seis_x = Sop_x * v.ravel()
seis_x = seis_x.reshape(nx, nt)

slope_y = -pylops.utils.signalprocessing.slope_estimate(y.T, dt, dx, smooth=6)[0]
Sop_y = pylops.signalprocessing.Seislet(slope_y.T, sampling=(dx, dt))

seis_y = Sop_y * y.ravel()
seis_y = seis_y.reshape(nx, nt)

slope_sum = -pylops.utils.signalprocessing.slope_estimate(sum.T, dt, dx, smooth=6)[0]
Sop_sum = pylops.signalprocessing.Seislet(slope_sum.T, sampling=(dx, dt))

seis_sum = Sop_sum * sum.ravel()
seis_sum = seis_sum.reshape(nx, nt)

print("P(x) + P(y)")
print(seis_x + seis_y)
print("P(x + y)")
print(seis_sum)

diff = abs((seis_x + seis_y) - seis_sum)
print("diff")
print(diff)

print(np.mean(diff),np.amax(diff),np.amin(diff))
