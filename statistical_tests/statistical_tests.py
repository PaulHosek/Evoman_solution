from scipy.stats import ttest_ind, wilcoxon
import numpy as np


def gen_ig_grid(file_name, nr_enemies = 8):
    """
    Generate Individual gain 2d array to compare 2 algorithms on 10 runs each.

    @param f_prefix_name:   name of the file up to, but not including the number or filetype.
                            Needs to be csv. Can include path.
                            example:"exp-selTournament_mutUniformInt_cxBlend_100_100_50_0.8_0.2_alg-eaMuPlusLambda_enemy-"

    @return: np.array of size 10, 8 (run, enemy).
    """
    grid_ig = np.zeros((10, nr_enemies))
    for i in range(1, nr_enemies+1):
        with open(file_name) as f:
            # idx, fitness, player, enemy, time
            cur = np.genfromtxt(f, delimiter=',')[1:]  # remove header row
            grid_ig[:, i - 1] = cur[:, 2] - cur[:, 3]
    return grid_ig




# gain values this time
cma_fname = "alg-CMA-ES_group-[1, 3, 4, 6, 7].txt"
mo_fname = "alg-MO-CMA-ES_group-[1, 3, 4, 6, 7].txt"








# t tests for each enemy
p_values = np.empty(8, dtype=float)
for enemy in range(8):
    p_values[enemy] = round(ttest_ind(cma_es[:, enemy], mo_cma_es[:, enemy])[1], 4)

print("P-values of independent samples t-tests for each enemy.")



# overall pairwise comparison for both alg means
# average individual gain per enemy
blend_means = np.mean(cma_es, axis=0)
sbx_means = np.mean(mo_cma_es, axis=0)

res_paried = wilcoxon(blend_means, sbx_means)
print("Non-parametric paired samples t-test results. Significant if at alpha = 0.05")
print(res_paried)


# ----------------------------------------------------------------------------------------------------------------------
