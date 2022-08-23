#!/usr/bin/python3
import h5py, numpy as np, uproot3
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
'''
collect all result and info into one h5 file
'''
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="ipt", nargs='+', help="input root file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-r", dest="result", nargs='+', help="fit result")
psr.add_argument('-b', dest='binwidth', help='binwidth')
psr.add_argument('-u', dest='up90', nargs='+', help='up90')
psr.add_argument('-s', dest='sf', nargs='+', help='shapefeature')
psr.add_argument('-m', dest='michel', nargs='+', help='michel info')
args = psr.parse_args()
print(args)
up90 = []
shapefeature = np.array([], dtype=[('t10080', '<f4'), ('over5090', '<f4')])
michelinfo = np.array([], dtype=[
    ('nMichel','<i2'), ('selectnMichel','<i2'), ('micheldistance','<f4'), ('nCap','<i2'), ('selectnCap','<i2'), ('ndistance','<f4'), ('MichelEdep','<f4'), ('PDNCEdep','<f4')])
for u9, sf, mi in zip(args.up90, args.sf, args.michel):
    with h5py.File(u9, 'r') as ipt:
        up90 = np.append(up90, ipt['t90'][:])
    with h5py.File(sf, 'r') as ipt:
        shapefeature = np.append(shapefeature, ipt['shapefeature'][:][['t10080','over5090']])
    with h5py.File(mi, 'r') as ipt:
        michelinfo = np.append(michelinfo, ipt['michel'][:])
r0List = uproot3.lazyarray(args.ipt, "evtinfo", "Edep_PromptR")
ncap = uproot3.lazyarray(args.ipt, "evtinfo", "CaptureTag")

hitTimeSingle = uproot3.lazyarray(args.ipt, "evtinfo", "HitTimeSingle", basketcache=uproot3.cache.ThreadSafeArrayCache("8 MB"))
entries = len(ncap)
fitRes = np.array([], dtype=[('eid', '<i4'), ('likelihood', '<f4'), ('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4'), ('t1', '<f4'), ('t2', '<f4'), ('t3', '<f4'), ('npeak', '<i2'),  ('Qedep', '<f4'), ('bkg', '<f4'), ('chisquare1', '<f4'), ('chisquare2', '<f4'),('chi1','<f4'),('chi2','<f4'),('ndf','<i2')])
mergeinfo = np.zeros(entries, dtype=[
    ('eid', '<i4'), ('likelihood', '<f4'), ('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4'), ('E1Norm', '<f4'), ('E2Norm', '<f4'), ('E3Norm', '<f4'), ('t1', '<f4'), ('t2', '<f4'), ('t3', '<f4'), ('npeak', '<i2'),  ('Qedep', '<f4'), ('bkg', '<f4'), ('chisquare1', '<f4'), ('chisquare2', '<f4'),('chi1','<f4'),('chi2','<f4'),('ndf','<i2'), ('Up90', 'f4'), ('t10080', '<f4'), ('over5090', '<f4'), ('nCap','<i2'), ('nMichel', '<i2'),('michelDist', '<f4'),('edepR','<f4'), ('nDist','<f4')
    ])
for rs in args.result:
    with h5py.File(rs, 'r') as ipt:
        fitRes = np.append(fitRes, ipt['likelihood'][:])
mergeinfo['eid'] = fitRes['eid']
mergeinfo['likelihood'] = fitRes['likelihood']
mergeinfo['E1'] = fitRes['E1']
mergeinfo['E2'] = fitRes['E2']
mergeinfo['E3'] = fitRes['E3']
fitResEtotal = fitRes['E1']+fitRes['E2']+fitRes['E3']
index = fitResEtotal!=0
mergeinfo['E1Norm'][index] = fitRes['E1'][index]/fitResEtotal[index]*fitRes['Qedep'][index]
mergeinfo['E2Norm'][index] = fitRes['E2'][index]/fitResEtotal[index]*fitRes['Qedep'][index]
mergeinfo['E3Norm'][index] = fitRes['E3'][index]/fitResEtotal[index]*fitRes['Qedep'][index]
mergeinfo['t1'] = fitRes['t1']
mergeinfo['t2'] = fitRes['t2']
mergeinfo['t3'] = fitRes['t3']
mergeinfo['npeak'] = fitRes['npeak']
mergeinfo['Qedep'] = fitRes['Qedep']
mergeinfo['bkg'] = fitRes['bkg']
mergeinfo['chisquare1'] = fitRes['chisquare1']
mergeinfo['chisquare2'] = fitRes['chisquare2']
mergeinfo['ndf'] = fitRes['ndf']
mergeinfo['chi1'] = fitRes['chi1']
mergeinfo['chi2'] = fitRes['chi2']
mergeinfo['Up90'] = up90
mergeinfo['t10080'] = shapefeature['t10080']
mergeinfo['over5090'] = shapefeature['over5090']

mergeinfo['nCap'] = michelinfo['selectnCap']
mergeinfo['nDist'] = michelinfo['ndistance']
mergeinfo['nMichel'] = michelinfo['nMichel']
mergeinfo['michelDist'] = michelinfo['micheldistance']
mergeinfo['edepR'] = r0List
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('evtinfo', data=mergeinfo, compression='gzip')
