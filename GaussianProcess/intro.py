import numpy as np
import matplotlib.pylab as plt

def exponential_cov(x, y, params):
  return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)
class GaussianKernel():
    def __init__(self,theta):
        self.theta=theta

    def __call__(self,X1,X2):
        X1=np.atleast_2d(X1)
        X2=np.atleast_2d(X2)
        temp1 = np.c_[np.sum(X1**2, axis=1)]
        temp2 = np.c_[np.sum(X2**2, axis=1)]
        norm = temp1 + temp2.T - 2*np.dot(X1, X2.T)
        return self.theta[0]*np.exp(-norm/theta[1])

theta=[1,10]
kernel=GaussianKernel(theta)
K=exponential_cov(0,0,theta)
xpts=np.arange(-3,3,0.01)
plt.errorbar(xpts,np.zeros(len(xpts)),yerr=K,capsize=0)
plt.show()