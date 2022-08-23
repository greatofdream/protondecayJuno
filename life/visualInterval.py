import uproot3
import argparse
import matplotlib.pyplot as plt
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest="interval", help="input interval root file")
psr.add_argument("-o", dest="opt", help="output")
args = psr.parse_args()
fcx = uproot3.lazyarray(args.interval, "fc", "observe")[0]
fcy = uproot3.lazyarray(args.interval, "fc", "up")[0]
bkg = uproot3.lazyarray(args.interval, "fc", "bkg")[0]
sensitivity = uproot3.lazyarray(args.interval, "fc", "sensitivity")[0]
print("upperlimit,sensitivity")
print("{};{}".format(fcy[0], sensitivity))
'''
bax = uproot3.lazyarray(args.interval, "bayes", "observe")[0]
bay = uproot3.lazyarray(args.interval, "bayes", "up")[0]
clx = uproot3.lazyarray(args.interval, "CLs", "observe")[0]
cly = uproot3.lazyarray(args.interval, "CLs", "up")[0]
'''
fig, ax = plt.subplots()
ax.plot(fcx[0:3000], fcy[0:3000], label='Feldman Cousin Method')
'''
ax.plot(bax[0:10000], bay[0:10000], label='Bayes Method')
ax.plot(clx[1:10000], cly[1:10000], label='CLs Method')
'''
ax.axhline(sensitivity,0,10,linestyle='--',label='sensitivity')
#ax.text(0.5,0.95, 'sensitivity for bkg {}:{:.2f}'.format(bkg,sensitivity), verticalalignment='top')
ax.set_title('observe - upper limits')
ax.set_xlabel('observe')
ax.set_ylabel('upperlimits')
ax.legend()
plt.savefig(args.opt)