import os
import numpy as np

enemies = ['3-6-7-2-4']
mu = [150]
lmbd = [150]
ngen = [100]
nrun = '10'
strategies = ['cma', 'cma-opl', 'cma-mo']

cnt = 0
for strategy in strategies:
    for enemy in enemies:
        for m in mu:
            for l in lmbd:
                for n in ngen:
                    cnt += 1
                    print(f"Run {cnt}")
                    print(f"Executing.. Strategy={strategy} enemies={enemy} mu={m} lambda={l} ngen={n}")
                    os.environ['mu'] = str(m)
                    os.environ['lambda'] = str(l)
                    os.environ['ngen'] = str(n)
                    os.environ['enemy'] = str(enemy)
                    os.system("python task2.py")
print(f"\n\n{cnt} experiments run")