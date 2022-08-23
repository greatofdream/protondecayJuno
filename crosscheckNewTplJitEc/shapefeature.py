import uproot3, numpy as np, argparse, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gc
'''
t90 whole use global maximum value as peak.
up100/up80<2 fw5m/fw9m<25
'''
def upTimeGet(ht, dt, delta=0, percent=1):
    smoothCount = np.zeros(ht.shape)
    for i in range(delta, len(ht)-delta):
        smoothCount[i] = np.mean(ht[(i-delta):(i+delta+1)])
    maxPeakIndex = np.argmax(smoothCount)
    maxPeak = smoothCount[maxPeakIndex]
    flag = 0
    t1 = 0
    # just set a default big value to avoid extreme event. 
    for i in range(delta, maxPeakIndex+1):
        if smoothCount[i+1] > percent*maxPeak and smoothCount[i]<= percent*maxPeak:
            t1 = i
            break
    return t1*dt
def downTimeGet(ht, dt, delta=0, percent=1):
    smoothCount = np.zeros(ht.shape)
    for i in range(delta, len(ht)-delta):
        smoothCount[i] = np.mean(ht[(i-delta):(i+delta+1)])
    maxPeak = np.max(smoothCount)
    maxPeakIndex = np.where(smoothCount==maxPeak)[0][-1]
    flag = 0
    t1 = 0
    # just set a default big value to avoid extreme event. 
    for i in range(maxPeakIndex-1, delta, -1):
        if smoothCount[i] <= percent*maxPeak and smoothCount[i+1]> percent*maxPeak:
            t1 = i
            break
    return t1*dt
def overTimeGet(ht, dt, delta=0, percent=1):
    smoothCount = np.zeros(ht.shape)
    for i in range(delta, len(ht)-delta):
        smoothCount[i] = np.mean(ht[(i-delta):(i+delta+1)])
    maxPeakIndex = np.argmax(smoothCount)
    maxPeak = smoothCount[maxPeakIndex]
    return np.where(smoothCount>=(percent*maxPeak))[0].shape[0]
def onePerCut(ht, timelength=500):
    # use 1% percent cut 500ns
    peak = np.max(ht)
    for i in range(100, len(ht)):
        if ht[i]<= 0.01*peak and ht[i+1]>0.01*peak:
            if (i-1+timelength)>len(ht):
                return ht[(len(ht)-timelength):len(ht)]
            else:
                return ht[(i-1):(i-1+timelength)]
if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-p', dest="pd", nargs='+', help="input protondecay root file")
    psr.add_argument("-o", dest="opt", help="output")
    psr.add_argument('-l', dest="log", help='whether direct log the result', default=False, type=bool)
    psr.add_argument("-d", dest="deltat", help="deltat of the smooth area", type=int)
    psr.add_argument("-c", dest="cut", help="cut of timelength", type=int, default=-1)
    args = psr.parse_args()
    print(args)
    if args.cut!=-1:
        print('cut:{}'.format(args.cut))
    iptFiles = args.pd
    output = args.opt
    evtID = uproot3.lazyarray(iptFiles, "evtinfo", "evtID")
    hitTimeSingle = uproot3.lazyarray(iptFiles, "evtinfo", "HitTimeSingle", basketcache=uproot3.cache.ThreadSafeArrayCache("8 MB"))
    entries=len(hitTimeSingle)
    print('{};{}'.format(hitTimeSingle.shape, entries))
    if not bool(args.log):
        print('plot and write h5')
        pdf=PdfPages(args.opt.replace('.h5', '.pdf'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        shapefeature = np.zeros(entries, dtype=[('t80', '<i2'), ('t100', '<i2'), ('over50', '<i2'), ('over90', '<i2'),('t10080', '<f4'), ('over5090', '<f4')])

        for i, ht in enumerate(hitTimeSingle):
            hc, hbins = np.histogram(ht, bins=1000, range=(0, 1000))
            # smooth the wave and get the num and hit
            # check the data
            if (np.where(ht==-1)[0].shape[0])==29873:
                print("all time is -1 in evtid:{}".format(evtID[i]))
                t0 = 0
            elif args.cut!=-1:
                hc = onePerCut(hc, args.cut)
                t20 = upTimeGet(hc, hbins[1]-hbins[0], args.deltat, 0.2)
                t80 = downTimeGet(hc, hbins[1]-hbins[0], args.deltat, 0.8)
                if t20>=t80:
                    print(t20,t80)
                    print(hc)
                    t80 += 1
                maxPeak = np.max(hc)
                t100 = np.where(hc==maxPeak)[0][-1]
                #t100 = np.argmax(hc)
                over50 = overTimeGet(hc, hbins[1]-hbins[0], args.deltat, 0.5)
                over90 = overTimeGet(hc, hbins[1]-hbins[0], args.deltat, 0.9)
                shapefeature[i] = (t80-t20, t100, over50, over90, t100/(t80-t20), over50/over90)
        fig, ax = plt.subplots()
        ax.set_title('T80 and T100 distribution')
        ax.hist(shapefeature['t80'], bins=100, range=[0, 300], histtype='step',label='t80')
        ax.hist(shapefeature['t100'], bins=100, range=[0, 300], histtype='step',label='t100')
        ax.set_xlabel('t/ns')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_title('over50 and over90 distribution')
        ax.hist(shapefeature['over50'], bins=100, range=[0, 300], histtype='step',label='over50')
        ax.hist(shapefeature['over90'], bins=100, range=[0, 300], histtype='step',label='over90')
        ax.set_xlabel('over threshold bin range')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_title('T100/T80 distribution')
        ax.hist(shapefeature['t10080'], bins=100, range=[0,20], histtype='step',label='w/o energy cut')
        ax.set_xlabel('ratio')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.set_title('over50/over90 distribution')
        ax.hist(shapefeature['over5090'], bins=100, range=[0,50], histtype='step',label='w/o energy cut')
        ax.set_xlabel('ratio')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()
        pdf.close()

        with h5py.File(output, 'w') as opt:
            opt.create_dataset('shapefeature', data=shapefeature, compression='gzip')
    else:
        with h5py.File(output, 'r') as ipt:
            t90 = ipt['shapefeature'][:]