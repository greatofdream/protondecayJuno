import uproot, numpy as np
import argparse
import h5py
N_mctotol, N_mc44yr = 400000, 159992
eids = [146891]
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="ipt", nargs='+', help="input root file")
psr.add_argument('--weight', dest='weight', help='genie weight fiel')
psr.add_argument("-o", dest="opt", help="output")
args = psr.parse_args()
# uproot.concatenate([i+':events' for i in args.ipt], filter_name='',library='np')['Waveform']
with uproot.open(args.weight) as ipt:
    weights = ipt["events/weight"].array(library='np')

nbkg = np.sum(weights[eids]) * N_mctotol/N_mc44yr
print(weights[eids], nbkg)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('weights', data=weights, compression='gzip')
    opt.attrs['nbkg'] = nbkg