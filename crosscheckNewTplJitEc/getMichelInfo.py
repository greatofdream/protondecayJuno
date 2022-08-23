import uproot3, numpy as np, argparse, h5py
from numba import jit
import matplotlib.pyplot as plt
'''
statistic of michel electron position between center of energy depoiste; michel number; neutron info
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
michelT = uproot3.lazyarray(iptFiles, "evtinfo", "MichelStartT")
michelX = uproot3.lazyarray(iptFiles, "evtinfo", "MichelStartX")
michelY = uproot3.lazyarray(iptFiles, "evtinfo", "MichelStartY")
michelZ = uproot3.lazyarray(iptFiles, "evtinfo", "MichelStartZ")
michelE = uproot3.lazyarray(iptFiles, "evtinfo", "MichelStartE")
MichelEdep = uproot3.lazyarray(iptFiles, "evtinfo", "MichelEdep")
x0List = uproot3.lazyarray(iptFiles, "evtinfo", "Edep_PromptX")
y0List = uproot3.lazyarray(iptFiles, "evtinfo", "Edep_PromptY")
z0List = uproot3.lazyarray(iptFiles, "evtinfo", "Edep_PromptZ")

PDNCT = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCStartT")
PDNCX = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCStartX")
PDNCY = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCStartY")
PDNCZ = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCStartZ")
PDNCE = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCStartE")
PDNCEdep = uproot3.lazyarray(iptFiles, "evtinfo", "PDNCEdep")

entries = len(michelT)
michelInfo = np.zeros(entries,dtype=[('nMichel','<i2'),('selectnMichel','<i2'),('micheldistance','<f4'),('nCap','<i2'),('selectnCap','<i2'),('ndistance','<f4'),('MichelEdep','<f4'),('PDNCEdep','<f4')])

for i,(t,x,y,z,x0,y0,z0,em,tn,xn,yn,zn,e,med,ned) in enumerate(zip(michelT, michelX, michelY, michelZ,x0List,y0List,z0List,michelE,PDNCT,PDNCX,PDNCY,PDNCZ,PDNCE,MichelEdep,PDNCEdep)):
    index = (t>=args.begin)&(t<=args.end)&(em>10)&(em<54)# 10MeV~54MeV, <1.5m
    michelInfo[i]['selectnMichel'],michelInfo[i]['micheldistance'] = michelStatistic(x[index],y[index],z[index],x0,y0,z0)
    michelInfo[i]['nMichel'] = len(x)
    indexn = (tn>=1000)&(tn<=2500000)&(e<=2.5)&(e>=1.9)#<5
    michelInfo[i]['selectnCap'],michelInfo[i]['ndistance'] = michelStatistic(xn[indexn],yn[indexn],zn[indexn],x0,y0,z0)
    michelInfo[i]['nCap'] = len(xn)
    michelInfo[i]['MichelEdep'] = np.sum(med)
    michelInfo[i]['PDNCEdep'] = np.sum(ned)
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
