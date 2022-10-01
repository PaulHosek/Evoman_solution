from scipy.stats import ttest_ind, wilcoxon
import numpy as np


def gen_ig_grid(f_prefix_path):
    """
    Generate Individual gain 2d array to compare 2 algorithms on 10 runs each.

    @param f_prefix_name:   name of the file up to, but not including the number or filetype.
                            Needs to be csv. Can include path.
                            example:"exp-selTournament_mutUniformInt_cxBlend_100_100_50_0.8_0.2_alg-eaMuPlusLambda_enemy-"

    @return: np.array of size 10, 8 (run, enemy).
    """
    grid_ig = np.zeros((10, 8))
    for i in range(1, 9):
        with open(f_prefix_path + str(i) + ".csv") as f:
            cur = np.genfromtxt(f, delimiter=',')[1:]  # remove header row
            grid_ig[:, i - 1] = cur[:, 1] = cur[:, 2]
    return grid_ig


# generate data grids
blend_fname = "exp-selTournament_mutUniformInt_cxBlend_100_100_50_0.8_0.2_alg-eaMuPlusLambda_enemy-"
sbx_fname = "exp-selTournament_mutUniformInt_cxSimulatedBinary_100_100_50_0.8_0.2_alg-eaMuPlusLambda_enemy-"
blend_igs = gen_ig_grid(f_prefix_path=blend_fname)
sbx_igs = gen_ig_grid(f_prefix_path=sbx_fname)

# t tests for each enemy
p_values = np.empty(8, dtype=float)
for enemy in range(8):
    p_values[enemy] = round(ttest_ind(blend_igs[:, enemy], sbx_igs[:, enemy])[1], 4)

print("P-values of independent samples t-tests for each enemy.")
print(p_values, "\n")

# overall pairwise comparison for both alg means
# average individual gain per enemy
blend_means = np.mean(blend_igs, axis=0)
sbx_means = np.mean(sbx_igs, axis=0)

res_paried = wilcoxon(blend_means, sbx_means)
print("Non-parametric paired samples t-test results. Significant if at alpha = 0.05")
print(res_paried)

# for enemy in range(8):
#     mean_igs[enemy] = np.mean(blend_igs[:, enemy], sbx_igs[:, enemy])
# print(mean_igs)
