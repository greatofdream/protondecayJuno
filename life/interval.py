import ROOT, numpy as np
import argparse
import h5py
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="ipt", help="input root file")
psr.add_argument("-o", dest="opt", help="output")
args = psr.parse_args()
with h5py.File(args.ipt, 'r') as ipt:
    bkg = ipt.attrs['nbkg']
ROOT.gInterpreter.ProcessLine('#include "interval.C"')
ROOT.fcData(bkg, args.opt)