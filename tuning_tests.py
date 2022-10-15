import os
import numpy as np

#strategies = ["cma-mo", "cma-opl", "cma"]
#strategies = ["cma-opl", "cma"]
strategies = ["cma-mo"]
enemies = ['1-2-3-4-7', '7-8']
ngen = [50]
trials = [100]

cnt = 0
for strategy in strategies:
    for trial in trials:
        for enemy in enemies:
            for gen in ngen:
                cnt += 1
                print(f"Run {cnt}")
                print(f"Executing.. strategy={strategy} {trial}trials {gen}generations enemies={enemy}")
                os.environ['STRATEGY'] = strategy
                os.environ['NGEN'] = str(gen)
                os.environ['NTRIALS'] = str(trial)
                os.environ['enemy'] = enemy
                os.system("python task2_tuning.py")
print(cnt)