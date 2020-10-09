import numpy as np
from nptyping import NDArray

class RBFKernel():
    def __init__(self, variance: float,length_scale: float):
        self.variance=variance
        self.length_scale = length_scale

    def __call__(self, X1: NDArray[float], X2: NDArray[float]) -> NDArray[float]:
        temp1= np.c_[np.sum(X1**2,axis=1)]
        temp2 = np.c_[np.sum(X2**2, axis=1)]
        norm = temp1 + temp2.T - 2*np.dot(X1,X2.T)
        return self.variance * np.exp(-norm / (2*self.length_scale**2))