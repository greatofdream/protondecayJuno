import numpy as np, argparse
psr = argparse.ArgumentParser()
psr.add_argument('-g', dest='geo', help='geometry')
psr.add_argument('-o', dest='opt', help='output filename')
args = psr.parse_args()
r = 19500
spmtstart = 300000
spmtend = 329873
spmtPos = np.zeros(spmtend-spmtstart,dtype=[('id', '<i4'), ('x','<f4'),('y','<f4'),('z','<f4')])
with open(args.geo) as ipt:
    lines = ipt.readlines()
for i, line in enumerate(lines):
    theta = list(map(float,line.split()))
    for j in range(1,3):
        theta[j] = theta[j]/180*np.pi
    spmtPos[i] = (theta[0], r*np.sin(theta[1])*np.cos(theta[2]),r*np.sin(theta[1])*np.sin(theta[2]),r*np.cos(theta[1]))
    
np.savetxt(args.opt, spmtPos, '%d %.3f %.3f %.3f')