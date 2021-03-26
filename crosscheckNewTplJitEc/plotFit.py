import uproot3 as uproot, numpy as np, argparse, h5py
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wavefit import wavefit
def addTpl(wave, tpl, t0, A, tplLength, fitlength=500, zoom=1,azoom=True):
    end = np.int(np.ceil(t0)+tplLength*zoom)-1
    if end >fitlength:
        end = fitlength-1
    if zoom==1:
        percent = t0 - np.floor(t0)
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = ti - np.int(np.ceil(t0))
            # print('interB:{};percent:{}'.format(interpB, percent))
            wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A
    else:
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = np.int(np.floor((ti - t0)/zoom))
            percent = (ti - t0)/zoom-interpB
            # print('interB:{};percent:{}'.format(interpB, percent))
            if azoom:
                wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A/zoom
            else:
                wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A
    return wave
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

Ecorr = np.loadtxt('fitEc/corr.txt')
print(Ecorr)

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
eSelect = []
for ei, ht, kst, dc, mt, pt, r0, energy in zip(evtID,hitTimeSingle,kaonStopTime, decayChannel,michelT,pdncT,r0List,qedep):
    if ei in checkEventId:
        #print(ei)
        hitTimeSelect.append(ht)
        selectIndex.append(ei)
        kaonSTSelect.append(kst)
        decayCSelect.append(dc)
        michelSelect.append(mt)
        pdncSelect.append(pt)
        r0Select.append(r0)
        eSelect.append(energy)
print('selectindex is {}'.format(selectIndex))
props = dict(boxstyle='round', facecolor='wheat', alpha=0)
entries=len(checkEventId)
likelihoodList = np.zeros(entries, dtype=[('eid', '<i4'), ('likelihood', '<f4'), ('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4'), ('t1', '<f4'), ('t2', '<f4'), ('t3', '<f4'), ('npeak', '<i2'),  ('Qedep', '<f4'), ('bkg', '<f4'), ('chisquare1', '<f4'), ('chisquare2', '<f4'),('chi1','<f4'),('chi2','<f4'),('ndf','<i2')])
likelihoodList['Qedep'] = qedep[selectIndex]
fitResult = [{},{},{}]
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
pdf=PdfPages(args.opt.replace('.h5', '.pdf'))
print(tpl)
print(mutpl)
class inftyObj(object):
    def __init__(self):
        self.fun = 1000000
iObj = inftyObj()
for idex, (eid, energy, ht, r0) in enumerate(zip(selectIndex, eSelect, hitTimeSelect,r0Select)):
    tplraw, edge = np.histogram(ht, range=[0, 1000], bins=np.int(1000/binwidth))
    EcorrIndex = np.floor((r0/17700)**3*10).astype(int)
    if EcorrIndex>=10:
        EcorrIndex = 9
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
        if energy<380:
            fitResult[1] = peakFit.minimizeKmuMI()
            fitResult[2] = iObj
        elif energy>420:
            fitResult[1] = iObj
            fitResult[2] = peakPiFit.minimizeKpiMI()
        else:
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
        if energy<380:
            fitResult[1] = peakBFit.minimizeKmuMI()
            fitResult[2] = iObj
        elif energy>420:
            fitResult[1] = iObj
            fitResult[2] = peakPiBFit.minimizeKpiMI()
        else:
            fitResult[1] = peakBFit.minimizeKmuMI()
            fitResult[2] = peakPiBFit.minimizeKpiMI()
    # expectN not represent the number of peak
    expectN = np.argmin([f.fun for f in fitResult])
    # print(fitResult[0].success,fitResult[0].x, fitResult[0].fun)
    # print(fitResult[1].success,fitResult[1].x, fitResult[1].fun)
    # print(fitResult[2].success,fitResult[2].x,fitResult[2].fun)
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
            label = 'mu'
            n = 1
            p2Result = peakFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[idex]['chi2'] = peakFit.calLSChi(p2Result)
            kt = addTpl(np.zeros(chiFitLen),tpl*10,fitResult[1].x[0],fitResult[1].x[2],tpl.shape[0],chiFitLen)
            mut = addTpl(np.zeros(chiFitLen),mutpl*10,fitResult[1].x[1],fitResult[1].x[3],mutpl.shape[0],chiFitLen)
            print(label)
            print(kt)
            print(mut)
            print(fitResult[1])
        else:
            label = 'pi'
            n = 2
            p2Result = peakPiFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[idex]['chi2'] = peakPiFit.calLSChi(p2Result)
            kt = addTpl(np.zeros(chiFitLen),tpl*10,fitResult[2].x[0],fitResult[2].x[2],tpl.shape[0],chiFitLen)
            mut = addTpl(np.zeros(chiFitLen),pitpl*10,fitResult[2].x[1],fitResult[2].x[3],pitpl.shape[0],chiFitLen,fitResult[2].x[-1])
            print(label)
            print(kt)
            print(mut)
            print(fitResult[2])
        likelihoodList[idex]['chisquare1'] = anFit.calChi(fitResult[0].fun)
        likelihoodList[idex]['chisquare2'] = np.min([peakFit.calChi(fitResult[1].fun), peakPiFit.calChi(fitResult[2].fun)])
    else:
        likelihoodList[idex]['ndf'] = anBFit.hitList[anBFit.hitList!=0].shape[0]
        begin = anBFit.begin
        p1Result = anBFit.fitP1Result(fitResult[0].x)
        likelihoodList[idex]['chi1'] = anBFit.calLSChi(p1Result)
        if fitResult[1].fun<fitResult[2].fun:
            label = 'mu'
            n = 1
            p2Result = peakBFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[idex]['chi2'] = peakBFit.calLSChi(p2Result)
            kt = addTpl(np.zeros(chiFitLen),tplb*10,fitResult[1].x[0],fitResult[1].x[2],tplb.shape[0],chiFitLen)
            mut = addTpl(np.zeros(chiFitLen),mutplb*10,fitResult[1].x[1],fitResult[1].x[3],mutplb.shape[0],chiFitLen)
        else:
            label = 'pi'
            n = 2
            p2Result = peakPiBFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[idex]['chi2'] = peakPiBFit.calLSChi(p2Result)
            kt = addTpl(np.zeros(chiFitLen),tplb*10,fitResult[2].x[0],fitResult[2].x[2],tplb.shape[0],chiFitLen)
            mut = addTpl(np.zeros(chiFitLen),pitplb*10,fitResult[2].x[1],fitResult[2].x[3],pitplb.shape[0],chiFitLen,fitResult[2].x[-1])
        print(label)
        print(kt)
        print(mut)
        print(fitResult[2])
        likelihoodList[idex]['chisquare1'] = anBFit.calChi(fitResult[0].fun)
        likelihoodList[idex]['chisquare2'] = np.min([peakBFit.calChi(fitResult[1].fun), peakPiBFit.calChi(fitResult[2].fun)])
    print('chi1:{};chi2:{};ndf:{},chi1/chi2:{}'.format(likelihoodList[idex]['chi1'],likelihoodList[idex]['chi2'],likelihoodList[idex]['ndf'],likelihoodList[idex]['chi1']/likelihoodList[idex]['chi2']))
    print('chisquare1:{};chisquare2:{};chi1/chi2:{}'.format(likelihoodList[idex]['chisquare1'],likelihoodList[idex]['chisquare2'],likelihoodList[idex]['chisquare1']/likelihoodList[idex]['chisquare2']))
    fig, ax = plt.subplots()

    ax.hist(ht, range=[begin, begin+chiFitLen], bins=np.int(chiFitLen/binwidth),alpha=1, histtype='step', label='simulation data')
    ax.set_title('evtid{}'.format(eid))
    ax.text(0.05,0.95, '$chi^2_1/chi^2_2$:{:.2f};'.format(likelihoodList[idex]['chisquare1']/likelihoodList[idex]['chisquare2']), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    # plot the fit result
    # np.savez('checkA',anFit.hitList,p1Result,peakFit.fitP2Result(fitResult[1].x),peakPiFit.fitP2Result(fitResult[2].x))
    # exit(0)
    ax.plot(range(begin,begin+chiFitLen),p1Result,alpha=0.5, label='AN fit')
    ax.plot(range(begin,begin+chiFitLen),p2Result,alpha=0.5, label='$K^+$ {} fit'.format(label))
    ax.plot(range(begin,begin+chiFitLen),kt,alpha=0.7, label='$K^+$')
    ax.plot(range(begin,begin+chiFitLen), mut,alpha=0.7, label='{}'.format(label))
    # finish plot the fit result
    # ax.vlines([i+begin for i in fitResult[expectN].x[:(likelihoodList[idex]['npeak'])]], 0, 10,label='reconstruct time')
    # ax.axvline(kaonSTSelect[idex],color='g',label='kaonStop')
    #ax.vlines(pdncSelect[idex], 0, 10,color='r',label='pdnc')
    #ax.vlines(michelSelect[idex], 0, 10,color='b',label='michel')
    ax.set_xlim([begin, begin+chiFitLen])
    ax.set_xlabel('hitTime/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    if expectN<=1:
        E1c = fitResult[1].x[2]/Ecorr[EcorrIndex][0]
        E2c = fitResult[1].x[3]/Ecorr[EcorrIndex][1]
    else:
        E1c = fitResult[2].x[2]/Ecorr[EcorrIndex][2]
        E2c = fitResult[2].x[3]/Ecorr[EcorrIndex][3]

    E1 = energy*E1c/(E1c+E2c)
    E2 = energy*E2c/(E1c+E2c)
    ax.text(0.4,0.6, 
    '$\Delta T$:{:.2f}\n$E_1$:{:.2f};$E_2$:{:.2f}\n$\chi^2/ndf$:{:.1f}/{}'.format(
        fitResult[n].x[1]-fitResult[n].x[0],E1,E2,likelihoodList[idex]['chisquare2'],chiFitLen-6), transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    pdf.savefig()
    plt.close()
pdf.close()
