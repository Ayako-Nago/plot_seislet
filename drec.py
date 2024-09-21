import numpy as np
import matplotlib.pyplot as plt

import pylops

plt.close('all')

Sop = pylops.signalprocessing.Seislet(slope.T, sampling=(dx, dt))

seis = Sop * d.ravel()
drec = Sop.inverse(seis)

seis = seis.reshape(nx, nt)
drec = drec.reshape(nx, nt)

nlevels_max = int(np.log2(nx))
levels_size = np.flip(np.array([2 ** i for i in range(nlevels_max)]))
levels_cum = np.cumsum(levels_size)

plt.figure(figsize=(14, 5))
plt.imshow(seis.T, cmap='gray', vmin=-clip, vmax=clip)
for level in levels_cum:
    plt.axvline(level-0.5, color='w')
plt.title('Seislet transform')
plt.colorbar()
plt.axis('tight')