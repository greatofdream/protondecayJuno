import uproot3 as uproot, numpy as np, argparse, h5py
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wavefit import wavefit

'''
check some eventid in the dataset, fit the energy of each peak use tpl, and init the minimize start point reasonable.
'''
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="pd", nargs='+', help="input protondecay root file")
psr.add_argument("-e", dest="ipt", nargs="+", help="input the event number")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-t", dest="tpl", help="template")
psr.add_argument("-a", dest="antpl", help="atmosphere template", default="")
psr.add_argument("-m", dest="mutpl", help="muon template")
psr.add_argument("-p", dest="pitpl", help="pi template")
psr.add_argument('-b', dest='binwidth', help='binwidth')
psr.add_argument('-s', dest='scale', help='E scale')
psr.add_argument('-f', dest='chiFitLen', help='calculate the chisquare wavelength', default=500)
args = psr.parse_args()
print(args)
Escale=np.float(args.scale)
with h5py.File(args.tpl, 'r') as ipt:
    # manual add 6.4 to adjust the energy scale
    tpl = ipt['singleTpl'][:]/Escale
    tplb = ipt['boundaryTpl'][:]/Escale
    Eunit = np.int(ipt.attrs['energy'])
tplLength = tpl.shape[0]# 2ns:500/2=250
print('tplLength:{},Eunit: {}'.format(tplLength, Eunit))
with h5py.File(args.mutpl, 'r') as ipt:
    # manual add 6.4 to adjust the energy scale
    mutpl = ipt['singleTpl'][:]/np.int(ipt.attrs['energy'])
    mutplb = ipt['boundaryTpl'][:]/np.int(ipt.attrs['energy'])
with h5py.File(args.pitpl, 'r') as ipt:
    # manual add 6.4 to adjust the energy scale
    pitpl = ipt['singleTpl'][:]/np.int(ipt.attrs['energy'])
    pitplb = ipt['boundaryTpl'][:]/np.int(ipt.attrs['energy'])
if args.antpl=="":
    antpl = tpl[:]
    antplLength = tplLength
else:
    with h5py.File(args.antpl, 'r') as ipt:
        antpl = ipt['singleTpl'][:]
        antplb = ipt['boundaryTpl'][:]
    antplLength = antpl.shape[0]
binwidth = np.int(args.binwidth)
chiFitLen = np.int(args.chiFitLen)
checkEventId = np.array([int(id) for id in args.ipt])
z = 0
r0Boundary = 15716
#/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/Document/SPMTModel/SPMTPosition
# 300000 329873
#/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data
# 300000 325600

spmtstart = 300000
spmtend = 325600

iptFiles = args.pd
evtID = [i for i in uproot.lazyarray(iptFiles, "evtinfo", "evtID")]
print(checkEventId)
# print(evtID==54233) warning : 54233 is scalar not array.it return false
selectIndex = []
#np.array([np.where(r==evtID)[0][0] for r in checkEventId])
qedep = uproot.lazyarray(iptFiles, "evtinfo", "Qedep")
kaonStopTime = uproot.lazyarray(iptFiles, "evtinfo", "KaonStopTime")
r0List = uproot.lazyarray(args.pd, "evtinfo", "Edep_PromptR")

decayChannel = uproot.lazyarray(iptFiles, "evtinfo", "DecayChannel")
michelT = uproot.lazyarray(iptFiles, "evtinfo", "MichelStartT")
pdncT = uproot.lazyarray(iptFiles, "evtinfo", "PDNCStartT")
hitTimeSingle = uproot.lazyarray(iptFiles, "evtinfo", "HitTimeSingle", basketcache=uproot.cache.ThreadSafeArrayCache("8 MB"))
hitTimeSelect = []
kaonSTSelect = []
decayCSelect = []
michelSelect = []
pdncSelect = []
r0Select = []
for ei, ht, kst, dc, mt, pt, r0 in zip(evtID,hitTimeSingle,kaonStopTime, decayChannel,michelT,pdncT,r0List):
    if ei in checkEventId:
        #print(ei)
        hitTimeSelect.append(ht)
        selectIndex.append(ei)
        kaonSTSelect.append(kst)
        decayCSelect.append(dc)
        michelSelect.append(mt)
        pdncSelect.append(pt)
        r0Select.append(r0)
print('selectindex is {}'.format(selectIndex))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
entries=len(checkEventId)
likelihoodList = np.zeros(entries, dtype=[('eid', '<i4'), ('likelihood', '<f4'), ('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4'), ('t1', '<f4'), ('t2', '<f4'), ('t3', '<f4'), ('npeak', '<i2'),  ('Qedep', '<f4'), ('bkg', '<f4'), ('chisquare1', '<f4'), ('chisquare2', '<f4'),('chi1','<f4'),('chi2','<f4'),('ndf','<i2')])
likelihoodList['Qedep'] = qedep[selectIndex]
fitResult = [{},{},{}]
E1bound=(20,600)
E2bound=(50,600)
E3bound=(20, 600)
Tbound=(20, 950)
fun21=lambda x: x[1]-x[0]-5
fun32=lambda x: x[2]-x[1]-60
peakFit = wavefit(tpl, 1000, chiFitLen)
peakFit.setTpl(mutpl)
peakPiFit = wavefit(tpl, 1000, chiFitLen)
peakPiFit.setTpl(pitpl)
anFit = wavefit(antpl, 1000, chiFitLen)


peakBFit = wavefit(tplb, 1000, chiFitLen)
peakBFit.setTpl(mutplb)
peakPiBFit = wavefit(tplb, 1000, chiFitLen)
peakPiBFit.setTpl(pitplb)
anBFit = wavefit(antplb, 1000, chiFitLen)
#exit(1)
pdf=PdfPages(args.opt.replace('.h5', '.pdf'))
for idex, (eid, ht, r0) in enumerate(zip(selectIndex, hitTimeSelect,r0Select)):
    tplraw, edge = np.histogram(ht, range=[0, 1000], bins=np.int(1000/binwidth))
    peakFit.error=False
    peakPiFit.error=False
    anFit.error=False
    peakBFit.error=False
    peakPiBFit.error=False
    anBFit.error=False
    if r0<=r0Boundary:
        print('eid {}'.format(eid))
        peakFit.cutWave(evtID[eid], tplraw)
        if peakFit.error:
            likelihoodList[idex]['eid'] = evtID[eid]
            likelihoodList[idex]['npeak'] = 0
            continue
        peakPiFit.cutWave(evtID[eid], tplraw)
        anFit.cutWave(evtID[eid], tplraw)
        fitResult[0] = anFit.minimizeAnMI()
        fitResult[1] = peakFit.minimizeKmuMI()
        fitResult[2] = peakPiFit.minimizeKpiMI()
    else:
        print('eid {}'.format(eid))
        peakBFit.cutWave(evtID[eid], tplraw)
        if peakBFit.error:
            likelihoodList[idex]['eid'] = evtID[eid]
            likelihoodList[idex]['npeak'] = 0
            continue
        peakPiBFit.cutWave(evtID[eid], tplraw)
        anBFit.cutWave(evtID[eid], tplraw)
        fitResult[0] = anBFit.minimizeAnMI()
        fitResult[1] = peakBFit.minimizeKmuMI()
        fitResult[2] = peakPiBFit.minimizeKpiMI()
    # expectN not represent the number of peak
    expectN = np.argmin([f.fun for f in fitResult])
    print(fitResult[0].success,fitResult[0].x, fitResult[0].fun)
    print(fitResult[1].success,fitResult[1].x, fitResult[1].fun)
    print(fitResult[2].success,fitResult[2].x,fitResult[2].fun)
    #likelihoodList[idex]['eid'] = eid
    likelihoodList[idex]['likelihood'] = fitResult[expectN].fun 
    if expectN == 0:
        likelihoodList[idex]['npeak'] = 1
        likelihoodList[idex]['E1'] = fitResult[expectN].x[expectN+1]*Eunit
        likelihoodList[idex]['t1'] = fitResult[expectN].x[0]*binwidth
    else:
        likelihoodList[idex]['npeak'] = 2
        likelihoodList[idex]['E1'] = fitResult[expectN].x[1+1]*Eunit
        likelihoodList[idex]['E2'] = fitResult[expectN].x[1+2]*Eunit
        likelihoodList[idex]['t1'] = fitResult[expectN].x[0]*binwidth
        likelihoodList[idex]['t2'] = fitResult[expectN].x[1]*binwidth
    likelihoodList[idex]['bkg'] = fitResult[expectN].x[-1]
    # x is ndarray
    if r0<=r0Boundary:
        likelihoodList[idex]['ndf'] = anFit.hitList[anFit.hitList!=0].shape[0]
        begin = anFit.begin
        p1Result = anFit.fitP1Result(fitResult[0].x)
        likelihoodList[idex]['chi1'] = anFit.calLSChi(p1Result)
        if fitResult[1].fun<fitResult[2].fun:
            p2Result = peakFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[idex]['chi2'] = peakFit.calLSChi(p2Result)
        else:
            p2Result = peakPiFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[idex]['chi2'] = peakPiFit.calLSChi(p2Result)
        likelihoodList[idex]['chisquare1'] = anFit.calChi(fitResult[0].fun)
        likelihoodList[idex]['chisquare2'] = np.min([peakFit.calChi(fitResult[1].fun), peakPiFit.calChi(fitResult[2].fun)])
    else:
        likelihoodList[idex]['ndf'] = anBFit.hitList[anBFit.hitList!=0].shape[0]
        begin = anBFit.begin
        p1Result = anBFit.fitP1Result(fitResult[0].x)
        likelihoodList[idex]['chi1'] = anBFit.calLSChi(p1Result)
        if fitResult[1].fun<fitResult[2].fun:
            p2Result = peakBFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[idex]['chi2'] = peakBFit.calLSChi(p2Result)
        else:
            p2Result = peakPiBFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[idex]['chi2'] = peakPiBFit.calLSChi(p2Result)
        likelihoodList[idex]['chisquare1'] = anBFit.calChi(fitResult[0].fun)
        likelihoodList[idex]['chisquare2'] = np.min([peakBFit.calChi(fitResult[1].fun), peakPiBFit.calChi(fitResult[2].fun)])
    print('chi1:{};chi2:{};ndf:{},chi1/chi2:{}'.format(likelihoodList[idex]['chi1'],likelihoodList[idex]['chi2'],likelihoodList[idex]['ndf'],likelihoodList[idex]['chi1']/likelihoodList[idex]['chi2']))
    print('chisquare1:{};chisquare2:{};chi1/chi2:{}'.format(likelihoodList[idex]['chisquare1'],likelihoodList[idex]['chisquare2'],likelihoodList[idex]['chisquare1']/likelihoodList[idex]['chisquare2']))
    fig, ax = plt.subplots()
    
    ax.hist(ht, range=[begin, begin+500], bins=np.int(500/binwidth), histtype='step')
    ax.set_title('evtid{}'.format(eid))
    ax.text(0.05,0.95, 'chisqure1/chisquare2:{:.2f};'.format(likelihoodList[idex]['chisquare1']/likelihoodList[idex]['chisquare2']), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # plot the fit result
    # np.savez('checkA',anFit.hitList,p1Result,peakFit.fitP2Result(fitResult[1].x),peakPiFit.fitP2Result(fitResult[2].x))
    # exit(0)
    ax.plot(range(begin,begin+chiFitLen),p1Result,alpha=0.7, label='fit 1 peak')
    ax.plot(range(begin,begin+chiFitLen),p2Result,alpha=0.7, label='fit 2 peak')
    # finish plot the fit result
    ax.vlines([i+begin for i in fitResult[expectN].x[:(likelihoodList[idex]['npeak'])]], 0, 10,label='reconstruct time')
    ax.axvline(kaonSTSelect[idex],color='g',label='kaonStop')
    ax.vlines(pdncSelect[idex], 0, 10,color='r',label='pdnc')
    ax.vlines(michelSelect[idex], 0, 10,color='b',label='michel')
    ax.set_xlim([begin, begin+400])
    ax.set_xlabel('hitTime')
    ax.set_ylabel('Entries')
    ax.legend()
    pdf.savefig()
    plt.close()
pdf.close()
