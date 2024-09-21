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
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import pylops



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

