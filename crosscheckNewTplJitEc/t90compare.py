import uproot, numpy as np, argparse, h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest="ipt", nargs='+', help="input protondecay root file")
    psr.add_argument("-c", dest="compare", nargs='+', help="compare h5 file")
    psr.add_argument('-o', dest="opt", help="output pdf file")
    args = psr.parse_args()
    print(args)
    iptFiles = args.ipt
    cmpFiles = args.compare
    output = args.opt
    evtID = uproot.lazyarray(iptFiles, "evtinfo", "evtID")
    t90ipt = uproot.lazyarray(iptFiles, "evtinfo", "UpT90")
    hitTimeSingle = uproot.lazyarray(iptFiles, "evtinfo", "HitTimeSingle", basketcache=uproot.cache.ThreadSafeArrayCache("8 MB"))

    t90cmp = []
    for u9 in args.compare:
        with h5py.File(u9, 'r') as ipt:
            t90cmp = np.append(t90cmp, ipt['t90'][:])
    pdf = PdfPages(args.opt)
    fig, ax = plt.subplots()
    ax.set_title('T90 distribution compare')
    ax.hist(t90ipt, bins=80, range=[0,80], histtype='step',label='Guo')
    ax.hist(t90cmp, bins=80, range=[0,80], histtype='step',label='crosscheck')
    print(np.histogram(t90ipt, bins=80, range=[0,80])[0])
    print(np.histogram(t90cmp, bins=80, range=[0,80])[0])
    selectT = (t90ipt-1)!=t90cmp
    print(t90ipt[selectT].shape)
    print([np.int(i) for i in t90ipt[selectT&((t90ipt-2)!=t90cmp)]])
    print(t90cmp[selectT&((t90ipt-2)!=t90cmp)])
    # selectT = (t90ipt-2)!=t90cmp
    # print(t90ipt[selectT].shape)
    ax.set_xlabel('t/ns')
    ax.set_ylabel('entries')
    ax.legend()
    pdf.savefig()
    plt.close()
    selectT90 = selectT #t90cmp>=78
    print('t90>=79: guoyuhang:{},crosscheck:{}'.format(t90ipt[t90ipt>79].shape,t90cmp[t90cmp>79].shape))
    fig, ax = plt.subplots()
    ax.set_title('T90 distribution compare different hist2d')
    h = ax.hist2d(t90cmp[selectT90], t90ipt[selectT90]-t90cmp[selectT90], bins=[80, 10], range=[[0,80],[0,10]])
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('crosscheck')
    ax.set_ylabel('guo-crosscheck')
    ax.legend()
    pdf.savefig()
    plt.close()
    pdf.close()
    exit(0)
    for evtid, ht, t90, t90i in zip(evtID[selectT90], hitTimeSingle[selectT90], t90cmp[selectT90], t90ipt[selectT90]):
        fig, ax = plt.subplots()
        ax.set_title('outline point evtid:{} up90:{},{}'.format(evtid, t90, t90i))
        ax.hist(ht, bins=500, range=[200,700], histtype='step',label='hittime distribution')
        ax.set_xlabel('t/ns')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig()
        plt.close()
    
    pdf.close()
