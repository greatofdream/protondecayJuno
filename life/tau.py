import sys
import numpy as np
import h5py
T=10
N=6.75
# e=0.3633
# n=2.446
n = np.loadtxt(sys.argv[1], skiprows=1, delimiter=',')[1]
with h5py.File(sys.argv[2], 'r') as ipt:
    e = ipt['eff'][:]['ratio'][-1]
print(n, e)
print('{} 10^33'.format(T*N*e/n))