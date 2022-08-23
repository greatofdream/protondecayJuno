import uproot, numpy as np, argparse, h5py
from numba import jit
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="ipt", help="input protondecay root file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-p", dest="pdg", nargs='+', help="pdgid list")
args = psr.parse_args()
storePdg = np.zeros((9530,),dtype=[('eid','<i4'),('pdg','<i4')])
@jit(nopython=True)
def arrin(u,v):
    # if u in v
    n = 0
    k = 0
    for j in range(v.shape[0]):
        for ki in range(k, u.shape[0]):
            if u[ki]==v[j]:
                n +=1
                k = ki
                break
            elif u[ki]>v[j]:
                k = ki
                break
        if n==u.shape[0]:
            return True
    return False

pdglist = np.array([np.int(i) for i in args.pdg])
Ncan = len(pdglist)
pdg = uproot.lazyarray(args.ipt, "events", "pdg", basketcache=uproot.cache.ThreadSafeArrayCache("8 MB"))
print('entries:{}'.format(len(pdg)))
begin = 0
for i, p in enumerate(pdg):
    p_u = np.unique(p)
    if p_u.shape[0]>=Ncan:
        if arrin(pdglist,p_u):
            print(i)
            storePdg[begin:(begin+len(p))]['eid'] = i
            storePdg[begin:(begin+len(p))]['pdg'] = p
            begin += len(p)
storePdg = storePdg[:begin]
with h5py.File(args.opt,'w') as opt:
    opt.create_dataset('pdg',data=storePdg, compression='gzip')

            # print(p)
