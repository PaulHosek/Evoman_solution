from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
import numpy as np


cma_fname_1_7 = np.loadtxt("alg-CMA-ES_group-[1, 3, 4, 6, 7].txt")
mo_fname_1_7 = np.loadtxt("alg-MO-CMA-ES_group-[1, 3, 4, 6, 7].txt")

cma_fname_7_8 = np.loadtxt("alg-CMA-ES_group-[7, 8].txt")
mo_fname_7_8 = np.loadtxt("alg-MO-CMA-ES_group-[7, 8].txt")


res_1_7 = wilcoxon( mo_fname_1_7, cma_fname_1_7, alternative = 'greater')
res_7_8 = wilcoxon( mo_fname_7_8,cma_fname_7_8, alternative = 'greater')




print("1,3,4,6,7")
print(res_1_7)
print("7,8")
print(res_7_8)
