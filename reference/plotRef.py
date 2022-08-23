import h5py,numpy as np, argparse
import matplotlib.pyplot as plt
psr = argparse.ArgumentParser()
psr.add_argument('-p', dest="pd", nargs='+', help="input rreference file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-c", dest="cross", help="crosscheck file")
args = psr.parse_args()
def readTxt(txt):
    with open(txt) as ipt:
        data = np.array(list(map(float, ipt.readlines()))).reshape((-1,))
        print(data)
    return data

an = readTxt(args.pd[0])/350
K = readTxt(args.pd[1])/100
mu = readTxt(args.pd[2])/150
pi = readTxt(args.pd[3])/350
with h5py.File(args.cross) as ipt:
    cross = ipt['singleTpl'][:]
fig, ax = plt.subplots()
ax.set_title('different response template')
ax.plot(an[:200],label='AN')
ax.plot(K[:200], label='K')
ax.plot(mu[:200], label='mu')
ax.plot(pi[:200], label='pi')
ax.plot(cross[:200], linestyle='--', label='crosschek')
ax.set_xlabel('t/ns')
ax.set_ylabel('entries')
ax.legend()
fig.savefig(args.opt)
plt.close()
fig, ax = plt.subplots()
ax.set_title('different response template', size='x-large')
ax.plot(an[:200],label='AN')
ax.plot(K[:200], label='K')
ax.plot(mu[:200], label='mu')
ax.plot(pi[:200], label='pi')
ax.set_xlabel('t/ns', size='x-large')
ax.set_ylabel('entries', size='x-large')
ax.legend()
fig.savefig(args.opt.replace('check.png',"originTpl.png"))
plt.close()