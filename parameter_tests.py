import os
import numpy as np

selection = ['selTournament', 'selRandom', 'selRoulette']
mutation = ['mutShuffleIndexes', 'mutGaussian', 'mutFlipBit', 'mutUniformInt']
crossover = ['cxTwoPoint', 'cxUniform', 'cxSimulatedBinary']
mu = [100]
lmbd = [100]
ngen = [20]
cxpb = [0.8]

cnt = 0
for sel in selection:
    for mut in mutation:
        for cx in crossover:
            for m in mu:
                for l in lmbd:
                    for n in ngen:
                        for cp in cxpb:
                            cnt += 1
                            print(f"Run {cnt}")
                            print(f"Executing.. sel={sel} mut={mut} cx={cx} mu={m} lambda={l} ngen={n} cxpb={cp}")
                            os.environ['sel'] = sel
                            os.environ['mut'] = mut
                            os.environ['cx'] = cx
                            os.environ['mu'] = str(m)
                            os.environ['lambda'] = str(l)
                            os.environ['ngen'] = str(n)
                            os.environ['cxpb'] = str(cp)
                            os.system("python ea1_demo.py")


print(cnt)