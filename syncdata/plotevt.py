import uproot,numpy as np, argparse
import matplotlib.pyplot as plt
psr = argparse.ArgumentParser()
psr.add_argument('-p', dest="pd", help="input root file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-i", dest="eid", help="input eid", type=int)
args = psr.parse_args()
print(args)
evtid=uproot.lazyarray(args.pd, "evtinfo", "evtID")           
hitTimeSingle = uproot.lazyarray(args.pd, "evtinfo", "HitTimeSingle", basketcache=uproot.cache.ThreadSafeArrayCache("8 MB"))
hitmp = hitTimeSingle[np.where(evtid==args.eid)[0][0]]
def onePerCut(ht, timelength=500):
    # use 1% percent cut 500ns
    peak = np.max(ht)
    for i in range(100, len(ht)):
        if ht[i]<= 0.01*peak and ht[i+1]>0.01*peak:
            if (i-1+timelength)>len(ht):
                return len(ht)-timelength
            else:
                return i-1
histmp=np.histogram(hitmp, range=[0,1000],bins=1000)
hi = onePerCut(histmp[0], 300)
fig, ax = plt.subplots()
ax.set_title('hittime distribution eid:{}'.format(args.eid))
ax.hist(hitmp,range=[hi, hi+100], bins=100,histtype='step',label='hittime distribution')
ax.set_xlabel('t/ns')
ax.set_ylabel('entries')
ax.legend()
fig.savefig(args.opt)
plt.close()