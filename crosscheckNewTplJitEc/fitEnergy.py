import uproot3 as uproot, numpy as np, argparse, h5py
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wavefit import wavefit

'''
fit the energy of each peak use tpl, and init the minimize start point reasonable.
'''
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="pd", nargs='+', help="input protondecay root file")
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
z = 0
r0Boundary = 15716
#/junofs/users/junoprotondecay/guoyh/offlines/offline3745/simulation_result/Document/SPMTModel/SPMTPosition
# 300000 329873
#/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data
# 300000 325600

spmtstart = 300000
spmtend = 325600

iptFiles = args.pd
qedep = uproot.lazyarray(iptFiles, "evtinfo", "Qedep")
hitTimeSingle = uproot.lazyarray(iptFiles, "evtinfo", "HitTimeSingle", basketcache=uproot.cache.ThreadSafeArrayCache("8 MB"))
r0List = uproot.lazyarray(args.pd, "evtinfo", "Edep_PromptR")
evtID = uproot.lazyarray(iptFiles, "evtinfo", "evtID")
# index = np.where(evtID==5796)[0][0]
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
entries=len(hitTimeSingle)
likelihoodList = np.zeros(entries, dtype=[('eid', '<i4'), ('likelihood', '<f4'), ('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4'), ('t1', '<f4'), ('t2', '<f4'), ('t3', '<f4'), ('npeak', '<i2'),  ('Qedep', '<f4'), ('bkg', '<f4'), ('chisquare1', '<f4'), ('chisquare2', '<f4'),('chi1','<f4'),('chi2','<f4'),('ndf','<i2')])
likelihoodList['Qedep'] = qedep
fitResult = [{},{},{}]
E1bound=(20,600)
E2bound=(50,600)
E3bound=(20, 600)
Tbound=(20, 950)
fun21=lambda x: x[1]-x[0]-4
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
class inftyObj(object):
    def __init__(self):
        self.fun = 1000000
iObj = inftyObj()
for eid, (ht, r0,energy) in enumerate(zip(hitTimeSingle,r0List, qedep)):
    tplraw, edge = np.histogram(ht, range=[0, 1000], bins=np.int(1000/binwidth))
    peakFit.error= False
    peakPiFit.error=False
    anFit.error=False
    peakBFit.error=False
    peakPiBFit.error=False
    anBFit.error=False
    if r0<=r0Boundary:
        peakFit.cutWave(evtID[eid], tplraw)
        if peakFit.error:
            likelihoodList[eid]['eid'] = evtID[eid]
            likelihoodList[eid]['npeak'] = 0
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
        peakBFit.cutWave(evtID[eid], tplraw)
        if peakBFit.error:
            likelihoodList[eid]['eid'] = evtID[eid]
            likelihoodList[eid]['npeak'] = 0
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
    # print(fitResult[0].success,fitResult[0].x, fitResult[0].fun)
    # print(fitResult[1].success,fitResult[1].x, fitResult[1].fun)
    # print(fitResult[2].success,fitResult[2].x,fitResult[2].fun)
    expectN = np.argmin([f.fun for f in fitResult])
    likelihoodList[eid]['eid'] = evtID[eid]
    likelihoodList[eid]['likelihood'] =fitResult[expectN].fun 
    if expectN == 0:
        likelihoodList[eid]['npeak'] = 1
        likelihoodList[eid]['E1'] = fitResult[expectN].x[expectN+1]*Eunit
        likelihoodList[eid]['t1'] = fitResult[expectN].x[0]*binwidth
        likelihoodList[eid]['bkg'] = fitResult[expectN].x[-1]
    elif expectN == 1:
        likelihoodList[eid]['npeak'] = expectN+1
        likelihoodList[eid]['E1'] = fitResult[expectN].x[1+1]*Eunit
        likelihoodList[eid]['E2'] = fitResult[expectN].x[1+2]*Eunit
        likelihoodList[eid]['t1'] = fitResult[expectN].x[0]*binwidth
        likelihoodList[eid]['t2'] = fitResult[expectN].x[1]*binwidth
        likelihoodList[eid]['bkg'] = fitResult[expectN].x[-1]
    else:
        likelihoodList[eid]['npeak'] = expectN+1
        likelihoodList[eid]['E1'] = fitResult[expectN].x[1+1]*Eunit
        likelihoodList[eid]['E2'] = fitResult[expectN].x[1+2]*Eunit*fitResult[expectN].x[-1]
        likelihoodList[eid]['t1'] = fitResult[expectN].x[0]*binwidth
        likelihoodList[eid]['t2'] = fitResult[expectN].x[1]*binwidth
        likelihoodList[eid]['bkg'] = fitResult[expectN].x[-2]
    # x is ndarray
    if r0<=r0Boundary:
        likelihoodList[eid]['ndf'] = anFit.hitList[anFit.hitList!=0].shape[0]
        begin = anFit.begin
        p1Result = anFit.fitP1Result(fitResult[0].x)
        likelihoodList[eid]['chi1'] = anFit.calLSChi(p1Result)
        if fitResult[1].fun<fitResult[2].fun:
            p2Result = peakFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[eid]['chi2'] = peakFit.calLSChi(p2Result)
        else:
            p2Result = peakPiFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[eid]['chi2'] = peakPiFit.calLSChi(p2Result)
        likelihoodList[eid]['chisquare1'] = anFit.calChi(fitResult[0].fun)
        likelihoodList[eid]['chisquare2'] = np.min([peakFit.calChi(fitResult[1].fun), peakPiFit.calChi(fitResult[2].fun)])
    else:
        likelihoodList[eid]['ndf'] = anBFit.hitList[anBFit.hitList!=0].shape[0]
        begin = anBFit.begin
        p1Result = anBFit.fitP1Result(fitResult[0].x)
        likelihoodList[eid]['chi1'] = anBFit.calLSChi(p1Result)
        if fitResult[1].fun<fitResult[2].fun:
            p2Result = peakBFit.fitP2Result(np.append(fitResult[1].x,1))
            likelihoodList[eid]['chi2'] = peakBFit.calLSChi(p2Result)
        else:
            p2Result = peakPiBFit.fitP2Result(fitResult[2].x,False)
            likelihoodList[eid]['chi2'] = peakPiBFit.calLSChi(p2Result)
        likelihoodList[eid]['chisquare1'] = anBFit.calChi(fitResult[0].fun)
        likelihoodList[eid]['chisquare2'] = np.min([peakBFit.calChi(fitResult[1].fun), peakPiBFit.calChi(fitResult[2].fun)])
    # print('chi1:{};chi2:{};ndf:{},chi1/chi2:{}'.format(likelihoodList[eid]['chi1'],likelihoodList[eid]['chi2'],likelihoodList[eid]['ndf'],likelihoodList[eid]['chi1']/likelihoodList[eid]['chi2']))
    # print('chisquare1:{};chisquare2:{};chi1/chi2:{}'.format(likelihoodList[eid]['chisquare1'],likelihoodList[eid]['chisquare2'],likelihoodList[eid]['chisquare1']/likelihoodList[eid]['chisquare2']))
selectNum = len(likelihoodList['npeak'][(likelihoodList['npeak']>1)&(qedep<600)&(qedep>200)])
EselectNum = len(qedep[(qedep<600)&(qedep>200)])
print('unique result: {}'.format(np.unique(likelihoodList['npeak'][(qedep<600)&(qedep>200)])))
print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))

with h5py.File('{}'.format(args.opt), 'w') as opt:
    opt.create_dataset('likelihood', data=likelihoodList, compression='gzip')

pdf=PdfPages(args.opt.replace('.h5', '.pdf'))
fig, ax = plt.subplots()
ax.set_title('peakNum distribution')
ax.hist(likelihoodList['npeak'], bins=[0, 1, 2, 3, 4, 5], histtype='step',label='peakNum')
ax.hist(likelihoodList['npeak'][(qedep<600)&(qedep>200)], bins=[0, 1, 2, 3, 4, 5], histtype='step',label='peakNum with energy cut')
ax.set_xlabel('peakNum')
ax.set_ylabel('entries')
ax.legend()
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution')
h = ax.hist2d(likelihoodList['E1'], likelihoodList['E2'], range=[[0, 600],[0, 600]], bins=[120,120])
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution with energy cut')
h = ax.hist2d(likelihoodList['E1'][(qedep<600)&(qedep>200)], likelihoodList['E2'][(qedep<600)&(qedep>200)], range=[[0, 600],[0, 600]], bins=[120,120])
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('t1 distribution')
ax.hist(likelihoodList['t1'], range=[0, 60], bins=20, histtype='step',label='deltaT')
ax.hist(likelihoodList['t1'][(qedep<600)&(qedep>200)], range=[0, 60], bins=20, histtype='step',label='deltaT with energy cut')
ax.set_xlabel('t1/ns')
ax.set_ylabel('entries')
ax.legend()
pdf.savefig()
plt.close()
pdf.close()
