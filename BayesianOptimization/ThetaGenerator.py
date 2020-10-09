import numpy as np
from scipy.stats import multivariate_normal
from nptyping import NDArray
import sys

class ThetaGenerator():
    mu = []
    var = []
    def __init__(self, dim: int, noise_var: float):
        self.dim=dim
        self.noise_var=noise_var

    #事後平均、事後分散の計算
    def calc(self, phi: NDArray[float], y: NDArray[float]) -> NDArray[float]:
        A = np.dot(phi.T, phi) + self.noise_var * np.eye(self.dim)
        Ainv=np.linalg.inv(A)
        Ainv_phi_T = np.dot(Ainv, phi.T)
        self.mu = np.dot(Ainv_phi_T, y)
        self.var = self.noise_var * Ainv        

    def getTheta(self, num:int) -> NDArray[float]:
        theta = multivariate_normal.rvs(mean=self.mu.ravel(), cov=self.var, size=num) #num個関数出してみる
        return theta