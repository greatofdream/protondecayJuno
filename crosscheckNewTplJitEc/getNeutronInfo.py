import uproot, numpy as np, argparse, h5py
from numba import jit
import matplotlib.pyplot as plt
'''
statistic of neutron capture position between center of energy depoiste; michel number
'''

@jit(nopython=True)
def michelStatistic(x,y,z,x0,y0,z0):
    length = len(x)
    if length==0:
        return 0,0
    distanceSum = 0
    for i in range(length):
        distanceSum += np.sqrt((x[i]-x0)**2+(y[i]-y0)**2+(z[i]-z0)**2)
    return length, distanceSum/length

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input root file')
psr.add_argument('-o', dest='opt', help='output h5 file')
psr.add_argument('-b', dest='begin', type=int, help='begin time of neutron')
psr.add_argument('-e', dest='end', type=int, help='end time of neutron')
args = psr.parse_args()
iptFiles = args.ipt
michelT = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartT")
PDNCX = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartX")
PDNCY = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartY")
PDNCZ = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartZ")
x0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptX")
y0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptY")
z0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptZ")
PDNCEdep = uproot.lazyarray(iptFiles, "evtinfo", "PDNCSTARTE")

entries = len(michelT)
michelInfo = np.zeros(entries,dtype=[('nCapture','<i2'),('ndistance','<f4')])

for i,(t,x,y,z,x0,y0,z0,e) in enumerate(zip(michelT, michelX, michelY, michelZ,x0List,y0List,z0List,PDNCEdep)):
    index = (t>=args.begin)&(t<=args.end)&(e<=2.5)&(e>=1.9)
    michelInfo[i]['nMichel'],michelInfo[i]['micheldistance'] = michelStatistic(x[index],y[index],z[index],x0,y0,z0)

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('neutron',data=michelInfo, compression='gzip')

fig,ax = plt.subplots()
ax.hist(michelInfo['micheldistance'][michelInfo['micheldistance']>0],bins=100,label='micheldistacne')
ax.set_title('ndistance')
ax.set_xlabel('ndistance')
ax.legend()
plt.savefig(args.opt.replace('.h5','.png'))