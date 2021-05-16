import numpy as np
from scipy.stats import multivariate_normal
from nptyping import NDArray
from scipy.linalg import cholesky
import sys
import random

class ThetaGenerator():
    mu = []
    var = []
    def __init__(self, seed:int, dim: int, noise_var: float):
        self.seed =seed
        self.dim=dim
        self.noise_var=noise_var

    #事後平均、事後分散の計算
    def calc(self, phi: NDArray[float], y: NDArray[float]) -> NDArray[float]:
        A = np.dot(phi.T, phi) + self.noise_var * np.eye(self.dim)
        Ainv=np.linalg.inv(A)
        Ainv_phi_T = np.dot(Ainv, phi.T)
        self.mu = np.dot(Ainv_phi_T, y)
        self.var = self.noise_var * Ainv
    
    def calc_init(self, phi: NDArray[float]) -> NDArray[float]:
        self.mu = np.zeros(phi.shape[1]).reshape(phi.shape[1],1)
        self.var = np.eye(phi.shape[1])

    def getTheta(self, num: int) -> NDArray[float]:
        np.random.seed(1)
        L = np.linalg.cholesky(self.var)
        z = np.random.randn(self.var.shape[1],num)
        tmp = np.dot(L, z)
        
        theta = self.mu.ravel() + tmp.T
        return theta