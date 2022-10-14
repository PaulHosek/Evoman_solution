import os
import numpy as np

enemies = ['7-8', '1-3-4-6-7']
mu = {'cma':'20', 'cma-mo':'23'}
lmbd = {'cma':'50', 'cma-mo':'71'}
sigma = {'cma':'0.486', 'cma-mo':'0.224'}
#cmo: 10, 20, 0.234
ngen = ['300']
nrun = '10'
strategies = ['cma-mo', 'cma']

cnt = 0
for strategy in strategies:
    for enemy in enemies:
        if len(enemy) > 3:
            mu = {'cma': '23', 'cma-mo': '30'}
            lmbd = {'cma': '114', 'cma-mo': '130'}
            sigma = {'cma': '0.122', 'cma-mo': '0.19'}
        for n in ngen:
            cnt += 1
            print(f"Run {cnt}")
            print(f"Executing.. Strategy={strategy} enemies={enemy} mu={mu[strategy]} lambda={lmbd[strategy]} sigma={sigma[strategy]} ngen={n}")
            os.environ['strategy'] = strategy
            os.environ['enemy'] = str(enemy)
            os.environ['nrun'] = nrun
            os.environ['ngen'] = n
            os.environ['mu'] = mu[strategy]
            os.environ['lambda'] = lmbd[strategy]
            os.environ['sigma'] = sigma[strategy]
            os.system("python task2.py")
print(f"\n\n{cnt} experiments run")