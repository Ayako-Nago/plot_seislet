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


for i in range(100):
    rng = np.random.default_rng()
    v = rng.random(4096)
    y = rng.random(4096)

    nh, nt = 64,64
    dh, dt = 8, 0.004 #初期値
    h, t = np.arange(nh) * dh, np.arange(nt) * dt

    pxmax = 1e-2
    npx = nt
    px = np.linspace(0, pxmax, npx)

    v = v.reshape(npx,nt)
    y = y.reshape(npx,nt)


    sum = v + y




    
    # slope_x = -pylops.utils.signalprocessing.slope_estimate(v.T, dt, dx, smooth=6)[0]
    # Sop_x = pylops.signalprocessing.Seislet(slope_x.T, sampling=(dx, dt))
    RLop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True, kind="hyperbolic", interp=False, engine="numpy")
    radon_v = RLop * v

    # seis_x = Sop_x * v.ravel()
    # seis_x = seis_x.reshape(nx, nt)

    # slope_y = -pylops.utils.signalprocessing.slope_estimate(y.T, dt, dx, smooth=6)[0]
    # Sop_y = pylops.signalprocessing.Seislet(slope_y.T, sampling=(dx, dt))

    # seis_y = Sop_y * y.ravel()
    # seis_y = seis_y.reshape(nx, nt)
    RLop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True, kind="hyperbolic", interp=False, engine="numpy")
    radon_y = RLop * y

    # slope_sum = -pylops.utils.signalprocessing.slope_estimate(sum.T, dt, dx, smooth=6)[0]
    # Sop_sum = pylops.signalprocessing.Seislet(slope_sum.T, sampling=(dx, dt))

    # seis_sum = Sop_sum * sum.ravel()
    # seis_sum = seis_sum.reshape(nx, nt)
    RLop = pylops.signalprocessing.Radon2D(t, h, px, centeredh=True, kind="hyperbolic", interp=False, engine="numpy")
    radon_sum = RLop * sum

    #diff = abs((seis_x + seis_y) - seis_sum)
    diff = abs((radon_v + radon_y) - radon_sum)

    print(i," : ",'{:.10f}'.format(np.mean(diff)),'{:.10f}'.format(np.amax(diff)),'{:.10f}'.format(np.amin(diff)))
