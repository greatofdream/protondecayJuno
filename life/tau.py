import sys
import numpy as np
import h5py
T=10
N=6.75
# e=0.3633
# n=2.446
n_limit, n_sensitivity = np.loadtxt(sys.argv[1], skiprows=1, delimiter=',')
with h5py.File(sys.argv[2], 'r') as ipt:
    e = ipt['eff'][:]['ratio'][-1]
print(n_limit, n_sensitivity, e)
print('{} 10^33'.format(T*N*e/n_limit))
print('{} 10^33'.format(T*N*e/n_sensitivity))
