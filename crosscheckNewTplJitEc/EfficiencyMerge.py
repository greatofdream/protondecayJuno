import h5py, numpy as np
import argparse
import pandas as pd
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs='+', help='input sig/bkg h5 file')
psr.add_argument('-o', dest='opt', help='output h5 file')
args = psr.parse_args()
def loadh5(f):
    with h5py.File(f, 'r') as ipt:
        nm, ntag = ipt.attrs['nm'], ipt.attrs['ntag']
        eff = ipt['eff'][:]
    return pd.DataFrame({
        '{}_{}_num'.format(nm, ntag): eff['cut'],
        '{}_{}_ratio'.format(nm, ntag): eff['Ratio_cut']
        }), '{}_{}'.format(nm, ntag)
h5s = [loadh5(s) for s in args.ipt]
df = pd.concat([s[0] for s in h5s], axis=1)
df['sum'] = np.sum(df[[s[1]+'_num' for s in h5s]].values, axis=1)
df['ratio'] = np.sum(df[[s[1]+'_ratio' for s in h5s]].values, axis=1)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('eff', data=df.values, compression='gzip')
print(df)