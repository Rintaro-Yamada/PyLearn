#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

#ブラックボックス関数
def blackbox_func(x):
    return x*np.sin(x)

#獲得関数（UCB)
def acq_ucb(mean, sig, beta=1000):
    return np.argmax(mean + sig * np.sqrt(beta))

#獲得関数（EI)
def acq_EI(x_grid,mean,sig,y):
    noise=0.01
    min_y = np.min(y)
    EI = np.zeros(len(mean))
    ind = np.where(sig != 0)[0]
    mean = mean[ind]
    sig = sig[ind]
    Z = (min_y - mean-noise) / sig
    EI[ind] = sig * Z * np.array([norm.cdf(z) for z in Z]) + sig * np.array([norm.pdf(z) for z in Z])

    '''
    EIの描画
    '''
    fig2 = plt.figure()
    plt.plot(x_grid, EI, 'r:', label=u'EI')
    plt.plot(x_grid[np.argmax(EI)],np.max(EI), 'r.', markersize=10, label=u'next_Observations')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('EI')
    plt.legend(loc='upper left')
    plt.savefig('2EI%02d.pdf' % (i))
    plt.close()

    return np.argmax(EI)

'''
#獲得関数（MSE)
def acq_MSE(y,mean,sig):
    gamma=-sum([y*log2(y)
    return 0
'''

#描画
def plot(x, y, X, y_pred, sigma, title=""):
    fig = plt.figure()
    plt.plot(x, blackbox_func(x), 'r:', label=u'$blackbox func(x) = (6x-2)^2\,\sin(12x-4)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma,(y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #plt.ylim(-10, 20)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig('betabig_fig_ucb%02d.pdf' % (i))
    plt.close()
 
if __name__ == '__main__':
    
    # データxの取りうる範囲
    x_grid = np.atleast_2d(np.linspace(0, 10, 1000)).T
    
    # 初期値として x=1, 9 の 2 点の探索をしておく.
    X = np.atleast_2d([1., 9.]).T
    y = blackbox_func(X).ravel()
    print(y)
    n_iteration = 10
    for i in range(n_iteration):
        gp = GaussianProcessRegressor()
        gp.fit(X,y)
        # 事後分布を求める
        posterior_mean, posterior_sig = gp.predict(x_grid, return_std=True)
        # 目的関数を最大化する次のパラメータの選択
        #idx = acq_EI(x_grid,posterior_mean, posterior_sig, y)
        idx = acq_ucb(posterior_mean, posterior_sig)
        x_next = x_grid[idx]

        plot(x_grid, y, X, posterior_mean, posterior_sig, title='Iteration=%2d,  x_next = %f'%(i+2, x_next))
    
        # 更新
        X = np.atleast_2d([np.r_[X[:, 0], x_next]]).T
        y = np.r_[y, blackbox_func(x_next)]
        