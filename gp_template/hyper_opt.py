import numpy as np
from kernel_fn import RBFkernel

def hyper_opt(X_train, y_train, hyper_grid):
    x_len = len(X_train)
    length_scale = np.c_[np.linspace(0.01, 1.0, hyper_grid)]
    l_tensor = np.tile(length_scale, hyper_grid*x_len**2).reshape(hyper_grid,hyper_grid,x_len,x_len)
    variance = np.c_[np.linspace(0.1, 10.0, hyper_grid)]
    v_tensor = np.tile(np.tile(variance, x_len ** 2).flatten(), hyper_grid).reshape(hyper_grid, hyper_grid, x_len, x_len)
    norm = X_train ** 2 - 2 * np.dot(X_train, X_train.T) + X_train.T ** 2
    norm_tensor = np.tile(norm, hyper_grid ** 2).reshape(hyper_grid, hyper_grid, x_len, x_len)
    K = v_tensor * np.exp(-norm_tensor / (2 * l_tensor ** 2)) + 1.0e-4 * np.tile(np.eye(x_len).flatten(), hyper_grid ** 2).reshape(hyper_grid, hyper_grid, x_len, x_len)
    yudo = -np.log(np.linalg.det(K)) - np.dot(np.dot(y_train.T, np.linalg.inv(K)), y_train).reshape(hyper_grid, hyper_grid)
    l_index, v_index = np.unravel_index(np.argmax(yudo), yudo.shape)
    return length_scale[l_index], variance[v_index]


# 2重for文使うgrid_search

def opt(X_train, y_train, hyper_grid):
    length_scale = np.linspace(0.01, 1.0, hyper_grid)
    variance = np.linspace(0.1, 10.0, hyper_grid)
    norm = X_train ** 2 - 2 * np.dot(X_train, X_train.T) + X_train.T ** 2
    yudo_max = -np.inf
    v_index = 0
    l_index = 0
    for i in range(hyper_grid):
        for j in range(hyper_grid):
            K = variance[i] * np.exp(-norm/(2*length_scale[j]**2)) + 1.0e-4*np.eye(len(X_train))
            yudo = -np.log(np.linalg.det(K)) - np.dot(np.dot(y_train.T, np.linalg.inv(K)), y_train)
            if (yudo > yudo_max):
                l_index = j
                v_index = i
                yudo_max = yudo
    
    return length_scale[l_index], variance[v_index]