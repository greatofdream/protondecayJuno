import numpy as np, pandas as pd
import argparse
from scipy.stats import poisson
'''
example of feldman cousins
'''
psr = argparse.ArgumentParser()
psr.add_argument("-b", dest='bkg',default=3,type=float)
psr.add_argument("-m", dest='mu', default=0.5, type=float)
args = psr.parse_args()
b=args.bkg
mu=args.mu
def pdf(x, arg):
    b, mu = arg
    return poisson.pmf(x,b+mu)
df = pd.DataFrame(columns=['n', 'P(n|\mu)','\mu_{best}','P(n|\mu_{best})','R', 'rank', 'rankP', 'U.L.','central'])
df['n'] = pd.Series(range(0,12))
df['P(n|\mu)'] = df['n'].apply(lambda x: poisson.pmf(x, b+mu))
df['\mu_{best}'] = df['n'].apply(lambda x: np.maximum(x-b,0))
df['P(n|\mu_{best})'] = poisson.pmf(df['n'], b+df['\mu_{best}'])
df['R'] = df['P(n|\mu)']/df['P(n|\mu_{best})']
df['rank'][np.argsort(df['R'])] = 12 - np.array(range(12))
df['rankP'][np.argsort(df['rank'])] = np.cumsum(df['P(n|\mu)'][np.argsort(df['rank'])])
df['U.L.'] = np.cumsum(df['P(n|\mu)'])
df['central'] = 1-df['U.L.']
print(df)
