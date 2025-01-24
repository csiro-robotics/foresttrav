# MIT License
#
# Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# Queensland University of Technology (QUT)
# 
# Author: Fabio Ruetz
import torch
import torch.nn as nn

from odap_tm.models.PUVoxTrav import confidence_exp_function

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math

mean = 0.5
std  = 2.2**2

# x = np.linspace(mean - 3*std**2, mean + 3*std**2, 100)
x = np.linspace(0, 20)
y = confidence_exp_function(torch.from_numpy(x), mean=mean, std=std, k_sigma=2).numpy()
# plt.hist(y)
plt.plot(x, y)
plt.show()


