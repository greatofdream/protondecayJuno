import h5py,numpy as np, argparse
psr = argparse.ArgumentParser()
psr.add_argument('-p', dest="pd", help="input rreference file")
psr.add_argument('-b', dest="boundary", default='', help="input rreference boundary file")
psr.add_argument("-o", dest="opt", help="output")
psr.add_argument("-s", dest='scale', help='energy scale',type=int)
args = psr.parse_args()
def readTxt(txt):
    with open(txt, 'r') as ipt:
        data = np.array(list(map(float, ipt.readlines()))).reshape((-1,))
        print(data)
    return data

with h5py.File(args.opt, 'w') as opt:
    opt.attrs['energy'] = 1
    opt.create_dataset('singleTpl', data=readTxt(args.pd)/args.scale, compression='gzip')
    if args.boundary!='':
        opt.create_dataset('boundaryTpl', data=readTxt(args.boundary)/args.scale, compression='gzip')