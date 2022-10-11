#!/usr/bin/python3
import h5py,numpy as np, uproot3
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
'''
view the fit result accroding to the order,Evis,Michel,Capture,Up90,shape feature,chi,deltaT,e1-e2
'''
psr = argparse.ArgumentParser()
psr.add_argument('-p', dest="pd", nargs='+', help="input protondecay root file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-i", dest="ipt", help="input fit result")
args = psr.parse_args()
print(args)
with h5py.File(args.ipt) as ipt:
    fitRes = ipt['evtinfo'][:]

entries = fitRes.shape[0]
qedep1000 = fitRes['Qedep']
ncap = fitRes['nCap']
nmichel = fitRes['nMichel']
up90 = fitRes['Up90']
hitTimeSingle = uproot3.lazyarray(args.pd, "evtinfo", "HitTimeSingle", basketcache=uproot3.cache.ThreadSafeArrayCache("8 MB"))
evtID = uproot3.lazyarray(args.pd, "evtinfo", "evtID")
upt90 = uproot3.lazyarray(args.pd, "evtinfo", "UpT90")
r0 = uproot3.lazyarray(args.pd, "evtinfo", "Edep_PromptR")
fitStore = np.zeros(entries, dtype=[('E1', '<f4'), ('E2', '<f4'), ('E3', '<f4')])
fitStore['E1'] = fitRes['E1']
fitStore['E2'] = fitRes['E2']
fitStore['E3'] = fitRes['E3']

fitResEtotal = fitRes['E1']+fitRes['E2']+fitRes['E3']
fitRes['E1'] = fitRes['E1']/fitResEtotal*qedep1000
fitRes['E2'] = fitRes['E2']/fitResEtotal*qedep1000
fitRes['E3'] = fitRes['E3']/fitResEtotal*qedep1000
fitEc = fitRes[(qedep1000<600)&(qedep1000>200)]
fitNc = fitRes[fitRes['npeak']>1]
fitEcNc = fitEc[fitEc['npeak']>1]

fitTc = fitEcNc[(fitEcNc['t2']-fitEcNc['t1'])>8.5]# Attention:Tc already cut with npeak and E
fitTcE2E1 = fitTc[(fitTc['E2']>100)&(fitTc['E2']<440)&(fitTc['E1']>30)&(fitTc['E1']<200)]
selectIndex = (qedep1000<600)&(qedep1000>200)&(fitRes['npeak']>1)&((fitRes['t2']-fitRes['t1'])>8.5)&(fitRes['E2']>100)&(fitRes['E2']<440)&(fitRes['E1']>30)&(fitRes['E1']<200)&(ncap<=1)&(nmichel>0)&(up90>=12)
fitTcE2E1AC = fitRes[selectIndex]
fitTcE2E1ACchi = fitTcE2E1AC[fitTcE2E1AC['npeak']>=2]
print('total entries:{};Eenergy cut {},rate:{:.5f}'.format(entries, fitEc.shape[0], fitEc.shape[0]/entries))
print('no energy cut, peak cut total entries:{};npeak>=2 select {}, npeak>=2: {:.5f}'.format(fitNc.shape[0], fitEcNc.shape[0], fitEcNc.shape[0]/entries))

print('energy cut, peak cut total entries:{};npeak>=2 select {}, npeak>=2: {:.5f}'.format(fitNc.shape[0], fitEcNc.shape[0], fitEcNc.shape[0]/entries))

print('energy cut, peak cut, t cut total entries:{} rate: {:.5f}'.format(fitTc.shape[0], fitTc.shape[0]/entries))
selectEcRc = (qedep1000<600)&(qedep1000>200)&(r0<=17500)
EcRcShape = fitRes[selectEcRc].shape[0]
print('energy cut, R cut; total entries:{} rate: {:.5f}, relative rate: {:.5f}'.format(EcRcShape, EcRcShape/entries, EcRcShape/fitEc.shape[0]))
selectEcMc = selectEcRc&(nmichel>0)
EcMcShape = fitRes[selectEcMc].shape[0]
print('energy cut, michel cut; total entries:{} rate: {:.5f}, relative rate: {:.5f}'.format(EcMcShape, EcMcShape/entries, EcMcShape/EcRcShape))
selectEcMcNc = selectEcMc&(ncap<=1)
EcMcNcShape = fitRes[selectEcMcNc].shape[0]
print('energy cut, michel cut, capture cut; total entries:{} rate: {:.5f}, relative rate: {:.5f}'.format(EcMcNcShape, EcMcNcShape/entries, EcMcNcShape/EcMcShape))
selectEcMcNcU9c = selectEcMcNc&(up90>=12)
EcMcNcU9cShape = fitRes[selectEcMcNcU9c].shape[0]
print('energy cut, michel cut, capture cut, t90 cut; total entries:{} rate: {:.5f}, relative rate: {:.5f}'.format(EcMcNcU9cShape, EcMcNcU9cShape/entries, EcMcNcU9cShape/EcMcNcShape))
selectAc = selectEcMcNcU9c
pdf = PdfPages(args.opt)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
binwidth = 1
fig, ax = plt.subplots()
h = ax.hist2d(fitStore[selectAc]['E1'], fitStore[selectAc]['E2'], range=[[0,200],[0,200]], bins=[200, 200], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_xlabel('E2/MeV')
pdf.savefig()
plt.close()
fig, ax = plt.subplots()
h = ax.hist2d(fitRes[selectAc]['E1'], fitRes[selectAc]['E2'], range=[[0,200],[0,200]], bins=[200, 200], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_xlabel('E2/MeV')
pdf.savefig()
plt.close()
for htw, fr, ut90 in zip( hitTimeSingle[selectAc],fitRes[selectAc], upt90[selectAc]):
    fig, ax = plt.subplots()
    ax.hist(htw, range=[0, 1000], bins=np.int(1000/binwidth), histtype='step', label='origin')
    fitTime = [fr['t1']]
    fitE = [fr['E1']]
    if fr['t2']!=0:
        fitTime.append(fr['t2'])
        fitE.append(fr['E2'])
    if fr['t3']!=0:
        fitTime.append(fr['t3'])
        fitE.append(fr['E3'])
    ax.vlines(fitTime, 0, 10)
    ax.set_title('evtid{}'.format(fr['eid']))
    ax.text(0.05,0.95, '{}\n chisquare1/chisquare2:{:.2f}\n energy:{:.2f}\n fitE:{}\n u90:{},ut90:{}'.format(fitTime, fr['chisquare1']/fr['chisquare2'], fr['Qedep'], fitE, fr['Up90'], ut90), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.set_xlabel('hitTime')
    ax.set_ylabel('Entries')
    ax.legend()
    pdf.savefig()
    plt.close()

pdf.close()
'''
pdf = PdfPages(args.opt)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)


fig, ax = plt.subplots()
ax.hist(qedep1000, range=[0,600], bins=120, histtype='step', label='qedep1000')
ax.hist(fitRes['E1']+fitRes['E2']+fitRes['E3'], range=[0,600], bins=120, histtype='step', label='fit1000ns')
ax.hist(fitResEtotal, range=[0,600], bins=120, histtype='step', label='origin fit1000ns')
ax.set_title("energy distribution")
ax.set_xlabel('energy/MeV')
ax.set_ylabel('entries')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution')
h = ax.hist2d(fitRes[fitRes['npeak']>1]['E1'], fitRes[fitRes['npeak']>1]['E2'], range=[[0, 600],[0, 600]], bins=[200,200], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution with Tcut')
h = ax.hist2d(fitTc[fitTc['npeak']>1]['E1'], fitTc[fitTc['npeak']>1]['E2'], range=[[0, 600],[0, 600]], bins=[200,200], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('deltaT distribution')
ax.hist(fitEcNc['t2']-fitEcNc['t1'], range=[0, 60], bins=60, histtype='step',label='deltaT')
ax.set_xlabel('deltaT/ns')
ax.set_ylabel('entries')
ax.legend()
pdf.savefig()
plt.close()

# fitTcE2E1 = fitTc[(fitTc['E2']>100)&(fitTc['E2']<440)&(fitTc['E1']>30)&(fitTc['E1']<200)]
#fitTcE1E2 = fitTcE2E1[((fitTcE2E1['E2']+fitTc['E1']+fitTc['E3'])<190)&((fitTc['E2']+fitTc['E1']+fitTc['E3'])>50)]
fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution with Tcut 30<E1<200,100<E2<440')
h = ax.hist2d(fitTcE2E1[fitTcE2E1['npeak']>1]['E1'], fitTcE2E1[fitTcE2E1['npeak']>1]['E2'], range=[[0, 250],[0, 500]], bins=[250,500], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()



fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution with Tcut 30<E1<200,100<E2<440,(ncap<=1)&(nmichel>0)&(up90>13)')
h = ax.hist2d(fitTcE2E1AC['E1'], fitTcE2E1AC['E2'], range=[[0, 250],[0, 500]], bins=[250,500], cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('deltaT distribution with AC')
ax.hist(fitTcE2E1AC['t2']-fitTcE2E1AC['t1'], range=[0, 60], bins=60, histtype='step',label='deltaT')
ax.set_xlabel('deltaT/ns')
ax.set_ylabel('entries')
ax.legend()
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.hist(qedep1000[selectIndex], range=[0,600], bins=120, histtype='step', label='qedep1000')
ax.hist(fitResEtotal[selectIndex], range=[0,600], bins=120, histtype='step', label='origin fit1000ns')
ax.set_title("energy distribution with AC")
ax.set_xlabel('energy/MeV')
ax.set_ylabel('entries')
pdf.savefig()
plt.close()



fig, ax = plt.subplots()
ax.hist(fitTcE2E1ACchi['chisquare1']*994/996/fitTcE2E1ACchi['chisquare2'], range=[0, 20], bins=200, histtype='step', label='chisquare distribution')
ax.set_title("chisquare distribution with AC")
ax.set_xlabel('chisquare')
ax.set_ylabel('entries')
pdf.savefig()
plt.close()

pdf.close()

fitTcE2E1ACchi3 = fitTcE2E1ACchi[(fitTcE2E1ACchi['chisquare1']*494/496/fitTcE2E1ACchi['chisquare2'])>3]
fitTcE2E1ACchi1_5 = fitTcE2E1ACchi[(fitTcE2E1ACchi['chisquare1']*494/496/fitTcE2E1ACchi['chisquare2'])>1.5]
print('energy cut, peak cut, t cut, 30<E1<200,100<E2<440,with (ncap<=1)&(nmichel>0)&(up90>13); total entries:{} rate: {:.4f}'.format(fitTcE2E1AC.shape[0], fitTcE2E1AC.shape[0]/entries))
print('energy cut, peak cut, t cut, 30<E1<200,100<E2<440,with (ncap<=1)&(nmichel>0)&(up90>13),chisquare>3; total entries:{} rate: {:.4f}'.format(fitTcE2E1ACchi3.shape[0], fitTcE2E1ACchi3.shape[0]/entries))
print('energy cut, peak cut, t cut, 30<E1<200,100<E2<440,with (ncap<=1)&(nmichel>0)&(up90>13),chisquare>1.5; total entries:{} rate: {:.4f}'.format(fitTcE2E1ACchi1_5.shape[0], fitTcE2E1ACchi1_5.shape[0]/entries))
#print('energy cut, peak cut, t cut, 30<E1<200,100<E2<440,with (ncap<=1)&(nmichel>0)&(up90>13),chisquare<3; total entries:{} rate: {:.4f}'.format(fitTcE2E1AC.shape[0]-fitTcE2E1ACchi3.shape[0], (fitTcE2E1AC.shape[0]-fitTcE2E1ACchi3.shape[0])/entries))
'''

'''
print('energy cut, peak cut, t cut, E2+E2<200 total entries:{} rate: {:.4f}'.format(fitTcE1E2.shape[0], fitTcE1E2.shape[0]/entries))
print('energy cut, peak cut, t cut, E2+E2<200,E2>22 total entries:{} rate: {:.4f}'.format(fitTcE1E2[fitTcE1E2['E2']>22].shape[0], fitTcE1E2[fitTcE1E2['E2']>22].shape[0]/entries))
print('energy cut, peak cut, t cut, E2+E2<200,E2>22,E1>22 total entries:{} rate: {:.4f}'.format(fitTcE1E2[(fitTcE1E2['E2']>22)&(fitTcE1E2['E1']>22)].shape[0], fitTcE1E2[(fitTcE1E2['E2']>22)&(fitTcE1E2['E1']>22)].shape[0]/entries))

selectFit = fitTcE1E2[(fitTcE1E2['E2']>22)&(fitTcE1E2['E1']>22)]

fig, ax = plt.subplots()
ax.hist(selectFit['E1']+selectFit['E2']+selectFit['E3'], range=[0,600], bins=120, histtype='step', label='fit1000ns')
ax.set_title("energy distribution")
ax.set_xlabel('energy/MeV')
ax.set_ylabel('entries')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('E1-E2 distribution')
h = ax.hist2d(selectFit['E1'], selectFit['E2'], range=[[0, 300],[0, 300]], bins=[100,100])
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('E1/MeV')
ax.set_ylabel('E2/MeV')
pdf.savefig()
plt.close()

fig, ax = plt.subplots()
ax.set_title('deltaT distribution')
ax.hist(selectFit['t2']-selectFit['t1'], range=[0, 60], bins=60, histtype='step',label='deltaT')
ax.set_xlabel('deltaT/ns')
ax.set_ylabel('entries')
ax.legend()
pdf.savefig()
plt.close()
'''

#print('energy cut, peak cut, t cut, E2+E2<200,E2>22,E1>22,E2>E1 total entries:{} rate: {:.4f}'.format(selectFit[(selectFit['E2']>selectFit['E1'])].shape[0], selectFit[(selectFit['E2']>selectFit['E1'])].shape[0]/entries))
'''
print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))

print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(entries, selectNum, selectNum/entries, 1-selectNum/entries))
print('total entries:{};select {} npeak>=2: {:.4f};npeak=1: {:.4f}'.format(EselectNum, selectNum, selectNum/EselectNum, 1-selectNum/EselectNum))
'''
