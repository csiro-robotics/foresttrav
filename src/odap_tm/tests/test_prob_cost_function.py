import numpy as np
import torch as ts

""" Evaluate and compare distance metrics for the feature evaluation:
    1. L1 norm
    2. L2 norm
    3. Cosine norm
"""



a = np.array([[0.1, - 0.1, 5, 10], [0.5, 0.3, 2, 6]])
b = np.array([[0.2, 1.1, 5, 2], [0.4, -0.3, 1, -0.1]])
ta  = ts.tensor(a)
tb = ts.tensor(b)

c = np.linalg.norm(a-b, axis=1) 
    
tc = ts.linalg.norm(ta-ta, axis=1).square().mean()
td = ts.nn.functional.mse_loss(ta, ta)


te= ts.nn.functional.pairwise_distance(ta,tb).square().mean() / 4
tf = ts.nn.functional.cosine_similarity(ta, tb)
print(te-tf)
l = 1
