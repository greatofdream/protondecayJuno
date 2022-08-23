import uproot3, numpy as np, argparse, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gc
'''
t90 whole use global maximum value as peak.
'''
def getT90WholeWalk(hcount, dt, delta=1, pdf=None):
    # smooth the histogram
    smoothCount = np.zeros(hcount.shape)
    for i in range(delta, len(hcount)-delta):
        smoothCount[i] = np.mean(hcount[(i-delta):(i+delta+1)])
    maxPeak = np.max(smoothCount)
    maxPeakIndex = np.where(smoothCount==maxPeak)[0][-1]
    flag = 0
    t1 = 0
    # just set a default big value to avoid extreme event. 
    t9 = 79
    for i in range(delta, maxPeakIndex+1):
        if smoothCount[i+1] > 0.1*maxPeak and smoothCount[i]<= 0.1*maxPeak:
            t1 = i
            break
    for i in range(maxPeakIndex-1, delta, -1):
        if smoothCount[i]<=0.9*maxPeak and smoothCount[i+1]>0.9*maxPeak:
            t9 = i
            break

    if pdf:
        fig, ax = plt.subplots()
        ax.step(smoothBin,smoothCount,where='mid', color='b',label='smooth')
        ax.plot(smoothBin, hcount, color='k', alpha=0.5, label='origin wave')
        for p in peakPos:
            ax.vlines(p, 0, 10, label='estimate peak', color='g')
        ax.text(0.05,0.95, 'estimate peak:{}'.format(peakPos), verticalalignment='top')
        ax.set_title('smooth hitTime')
        ax.set_xlabel('hitTime/ns')
        ax.set_ylabel('entries')
        ax.set_xlim(left=0, right=1500)
        ax.legend()
        pdf.savefig()
        plt.close()
    return (t9-t1)*dt
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
    qedep = uproot3.lazyarray(iptFiles, "evtinfo", "Qedep")
    hitTimeSingle = uproot3.lazyarray(iptFiles, "evtinfo", "HitTimeSingle", basketcache=uproot3.cache.ThreadSafeArrayCache("8 MB"))
    entries=len(hitTimeSingle)
    print('{};{}'.format(hitTimeSingle.shape, entries))
    if not bool(args.log):
        print('plot and write h5')
        pdf=PdfPages(args.opt.replace('.h5', '.pdf'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        t90 = np.zeros((entries,))

        for i, ht in enumerate(hitTimeSingle):
            hc, hbins = np.histogram(ht, bins=1000, range=(0, 1000))
            # smooth the wave and get the num and hit
            # check the data
            if (np.where(ht==-1)[0].shape[0])==29873:
                print("all time is -1 in evtid:{}".format(evtID[i]))
                t0 = 0
            elif args.cut!=-1:
                hc = onePerCut(hc, args.cut)
                t0 = getT90WholeWalk(hc, hbins[1]-hbins[0], delta=args.deltat)
            t90[i] = t0
        fig, ax = plt.subplots()
        ax.set_title('T90 distribution')
        ax.hist(t90, bins=80, range=[0,80], histtype='step',label='w/o energy cut')
        ax.hist(t90[(qedep<600)&(qedep>200)], bins=80, range=[0,80], histtype='step',label='energy cut')
        ax.set_xlabel('t/ns')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()
        pdf.close()

        with h5py.File(output, 'w') as opt:
            opt.create_dataset('t90', data=t90, compression='gzip')
    else:
        with h5py.File(output, 'r') as ipt:
            t90 = ipt['t90'][:]
    EselectNum = len(qedep[(qedep<600)&(qedep>200)])

    selectNum = len(t90[(t90>=13)&(qedep<600)&(qedep>200)])
    print('total entries:{};select {} t90>=13: {:.4f};t90<13: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
    print('total entries:{};select {} t90>=13: {:.4f};t90<13: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))
    selectNum = len(t90[(t90>=9)&(qedep<600)&(qedep>200)])
    print('total entries:{};select {} t90>=9: {:.4f};t90<9: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
    print('total entries:{};select {} t90>=9: {:.4f};t90<9: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))
    selectNum = len(t90[(t90>=10)&(qedep<600)&(qedep>200)])
    print('total entries:{};select {} t90>=10: {:.4f};t90<10: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
    print('total entries:{};select {} t90>=10: {:.4f};t90<10: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))
    selectNum = len(t90[(t90>=11)&(qedep<600)&(qedep>200)])
    print('total entries:{};select {} t90>=11: {:.4f};t90<11: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
    print('total entries:{};select {} t90>=11: {:.4f};t90<11: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))
    selectNum = len(t90[(t90>=12)&(qedep<600)&(qedep>200)])
    print('total entries:{};select {} t90>=12: {:.4f};t90<12: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
    print('total entries:{};select {} t90>=12: {:.4f};t90<12: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))

