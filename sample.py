#http://statmodeling.hatenablog.com/entry/how-to-use-GPy

import GPy
import numpy as np
import matplotlib.pyplot as plt

kernel=GPy.kern.RBF(input_dim=1,variance=1,lengthscale=0.2)

np.random.seed(0)
N_sim=100
x_sim=np.linspace(-1,-1,N_sim)
