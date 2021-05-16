import numpy as np
from scipy.stats import norm, multivariate_normal
def ei(X):
    return 

def ucb(X_train, pred_mean, pred_var):
    return pred_mean + np.sqrt(2 * np.log(X_train.shape[0] ** 2 + 1)) * np.sqrt(pred_var)

def mes(y_star, pred_mu, pred_var, func_num, train_index):
    y_sample = np.tile(y_star, (1,pred_mu.shape[0])).T
    gamma_y = (y_sample - pred_mu) / np.sqrt(pred_var)
    psi_gamma = norm.pdf(gamma_y, loc=0, scale=1)
    large_psi_gamma = norm.cdf(gamma_y, loc=0, scale=1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_y*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma
    alpha = np.sum(temp, axis=1) / func_num

    return alpha