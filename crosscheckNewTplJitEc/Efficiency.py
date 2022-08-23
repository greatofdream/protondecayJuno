import h5py, numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import config
thresholds = config.thresholds
# color bar setting
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input sig/bkg h5 file')
psr.add_argument('-o', dest='opt', help='output h5 file')
psr.add_argument('--nm', dest='nm', type=int, help='number of michel')
psr.add_argument('--ntag', dest='ntag', type=int, help='number of neutron')

args = psr.parse_args()

with h5py.File(args.ipt,'r') as ipt:
    info = ipt['evtinfo'][:]
nm = args.nm
ntag = args.ntag
# thresholds: eDepR, QEdep, nCapture, closed interval
eR_l, eR_r = thresholds[nm, ntag][['eR_l', 'eR_r']]
qe_l, qe_r = thresholds[nm, ntag][['Qedep_l', 'Qedep_r']]
nC_l, nC_r = thresholds[nm, ntag][['nCap_l', 'nCap_r']]
mD_l, mD_r = thresholds[nm, ntag][['michelDist_l', 'michelDist_r']]
nD_l, nD_r = thresholds[nm, ntag][['neutronDist_l', 'neutronDist_r']]
np_l, np_r = thresholds[nm, ntag][['npeak_l', 'npeak_r']]
t12_l, t12_r = thresholds[nm, ntag][['t12_l', 't12_r']]
E2_1_l, E2_1_r = thresholds[nm, ntag][['E2_1_l', 'E2_1_r']]
E2_2_l, E2_2_r = thresholds[nm, ntag][['E2_2_l', 'E2_2_r']]
E1_l, E1_r = thresholds[nm, ntag][['E1_l', 'E1_r']]
chi12_l, chi12_r = thresholds[nm, ntag][['chi12_l', 'chi2_r']]
# recording the efficiency: cut, nm, cut & nm, R_cut, R_cut|nm
efficiency = np.zeros((9,), dtype=[
    ('cut', '<f4'), ('nm', '<f4'), ('cutAndnm', '<f4'), ('Ratio_cut', '<f4'), ('Ratio_cutInnm', '<f4')
    ])
print('cut, nm, cut & nm, R_cut, R_cut|nm')
cursor = 0
index = (info['Qedep']>qe_l)&(info['Qedep']<qe_r)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
index = index & (info['edepR']<eR_r)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
index = index & (info['nMichel']==nm)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
index = index & (info['michelDist']<=mD_r)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
index = index & (info['nCap']<=nC_r) & (info['nCap']>=nC_l)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
index_tmp = index & (info['nDist']<nD_r)
efficiency[cursor] = (np.sum(index_tmp), np.sum(info['nMichel']==nm), np.sum(index_tmp&(info['nMichel']==nm)), np.sum(index_tmp)/len(info), np.sum(index_tmp&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
## chi2
index = index_tmp & (info['chisquare1']>info['chisquare2']*chi12_l)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
## t12
index = index & ((info['t2'] - info['t1'])>t12_l)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
cursor += 1
## E1, E2
index = index&(((info['E2Norm']>E2_1_l)&(info['E2Norm']<E2_1_r))|((info['E2Norm']>E2_2_l)&(info['E2Norm']<E2_2_r)))&(info['E1Norm']<E1_r)&(info['E1Norm']>E1_l)
efficiency[cursor] = (np.sum(index), np.sum(info['nMichel']==nm), np.sum(index&(info['nMichel']==nm)), np.sum(index)/len(info), np.sum(index&(info['nMichel']==nm))/np.sum(info['nMichel']==nm))
print(efficiency)
with h5py.File(args.opt, 'w') as opt:
    opt.attrs['nm'] = nm
    opt.attrs['ntag'] = ntag
    opt.create_dataset('eff', data=efficiency, compression='gzip')
with PdfPages(args.opt+'.pdf') as pdf:    
    fig,ax = plt.subplots()
    h = ax.hist2d(info[index_tmp]['E1Norm'], info[index_tmp]['E2Norm'], bins=[100,100], cmap=cmap)
    fig.colorbar(h[3])
    ax.set_xlabel('E1Norm/MeV')
    ax.set_ylabel('E2Norm/MeV')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(info[index]['E1Norm'], info[index]['E2Norm'], bins=[100,100], cmap=cmap)
    fig.colorbar(h[3])
    ax.set_xlabel('E1Norm/MeV')
    ax.set_ylabel('E2Norm/MeV')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(info[index]['Up90'], info[index]['t2']-info[index]['t1'], bins=[100,100], cmap=cmap)
    ax.set_xlabel('up90/ns')
    ax.set_ylabel('$t_2-t_1$/ns')
    fig.colorbar(h[3])
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(info[index]['t2']-info[index]['t1'], bins=100, histtype='step')
    ax.set_title('FitTime distribution')
    ax.set_xlabel('{}/ns'.format('FitTime'))
    ax.set_ylabel('{}'.format('Entries'))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(info[index]['chi1']/info[index]['chi2']*(info[index]['ndf']-4)/(info[index]['ndf']-6),bins=100,range=[0,10])
    ax.set_title('chisquare12 distribution')
    ax.set_xlabel('{}'.format('chisquare'))
    ax.set_ylabel('{}'.format('Entries'))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist2d(info[index]['chi1']/info[index]['chi2'],info[index]['t2']-info[index]['t1'],bins=[100,100])
    ax.set_title('FitTime-chi12')
    ax.set_ylabel('FitTime/ns')
    ax.set_xlabel('chi12')
    pdf.savefig(fig)
