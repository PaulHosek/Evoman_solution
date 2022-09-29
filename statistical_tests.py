from scipy.stats import ttest_ind
from os import path
import numpy as np

script_dir = path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "EA1/data.txt"
cur_path = path.join(script_dir, rel_path)

with open(cur_path, r) as f:
    data = np.loadtxt(f)


res = ttest_ind(data,data2)[0]

