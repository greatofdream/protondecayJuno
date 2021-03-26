import uproot, numpy as np, argparse, h5py
from numba import jit
import matplotlib.pyplot as plt
'''
statistic of michel electron position between center of energy depoiste; michel number
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
psr.add_argument('-b', dest='begin', type=int, help='begin time of michel electron')
psr.add_argument('-e', dest='end', type=int, help='end time of michel electron')
args = psr.parse_args()
iptFiles = args.ipt
michelT = uproot.lazyarray(iptFiles, "evtinfo", "MichelStartT")
michelX = uproot.lazyarray(iptFiles, "evtinfo", "MichelStartX")
michelY = uproot.lazyarray(iptFiles, "evtinfo", "MichelStartY")
michelZ = uproot.lazyarray(iptFiles, "evtinfo", "MichelStartZ")
x0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptX")
y0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptY")
z0List = uproot.lazyarray(iptFiles, "evtinfo", "Edep_PromptZ")

PDNCT = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartT")
PDNCX = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartX")
PDNCY = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartY")
PDNCZ = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartZ")
PDNCE = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartE")

entries = len(michelT)
michelInfo = np.zeros(entries,dtype=[('nMichel','<i2'),('selectnMichel','<i2'),('micheldistance','<f4'),('nCap','<i2'),('selectnCap','<i2'),('ndistance','<f4')])

for i,(t,x,y,z,x0,y0,z0,tn,xn,yn,zn,e) in enumerate(zip(michelT, michelX, michelY, michelZ,x0List,y0List,z0List,PDNCT,PDNCX,PDNCY,PDNCZ,PDNCE)):
    index = (t>=args.begin)&(t<=args.end)# 10MeV~54MeV, <1.5m
    michelInfo[i]['selectnMichel'],michelInfo[i]['micheldistance'] = michelStatistic(x[index],y[index],z[index],x0,y0,z0)
    michelInfo[i]['nMichel'] = len(x)
    indexn = (tn>=1000)&(tn<=2500000)&(e<=2.5)&(e>=1.9)#<5
    michelInfo[i]['selectnCap'],michelInfo[i]['ndistance'] = michelStatistic(xn[indexn],yn[indexn],zn[indexn],x0,y0,z0)
    michelInfo[i]['nCap'] = len(xn)
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('michel',data=michelInfo, compression='gzip')

fig,ax = plt.subplots()
ax.hist(michelInfo['micheldistance'][michelInfo['micheldistance']>0],bins=100,label='micheldistacne')
ax.set_title('micheldistance')
ax.set_xlabel('micheldistance')
ax.legend()
plt.savefig(args.opt.replace('.h5','.png'))
fig,ax = plt.subplots()
ax.hist(michelInfo['ndistance'][michelInfo['ndistance']>0],bins=100,label='ndistacne')
ax.set_title('ndistance')
ax.set_xlabel('ndistance')
ax.legend()
plt.savefig(args.opt.replace('.h5','n.png'))