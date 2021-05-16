import matplotlib.pyplot as plt
import numpy as np
import sys

def predict_plot(X, y, X_train, y_train, pred_mean, pred_var, fig_name):
    plt.plot(X, y, "r--", label="true_fn")
    plt.plot(X_train, y_train, "ro", label="obeserved_point")
    plt.plot(X, pred_mean, "b", label="pred_mean")
    plt.fill_between(X.flatten(), (pred_mean + 1.98 * np.sqrt(pred_var)).flatten(), (pred_mean - 1.98 * np.sqrt(pred_var)).flatten(), alpha=0.3, color="blue", label="credible interval")
    plt.legend(loc='lower left')
    plt.savefig(fig_name)
    plt.close()

def rfm_fn_plot(X,y, f_x, pred_mean, pred_var, fig_name):
    func_num = f_x.shape[0]
    plt.plot(X, y, "r--", label="true_fn")
    plt.plot(X, pred_mean, "b", label="pred_mean")
    plt.fill_between(X.flatten(), (pred_mean + 1.98 * np.sqrt(pred_var)).flatten(), (pred_mean - 1.98 * np.sqrt(pred_var)).flatten(), alpha=0.3, color="blue", label="credible interval")
    for i in range(func_num-1):
        plt.plot(X, f_x[i], "g")
    plt.plot(X, f_x[func_num - 1], "g", label = "rfm_fn")
    plt.legend(loc='lower left')
    plt.savefig(fig_name)
    plt.close()