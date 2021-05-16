import numpy as np
import sys
from ThetaGenerator import ThetaGenerator
from sklearn.utils import check_array, check_random_state, as_float_array

def RFM(x, dim, omega, b, variance):
    phi = np.sqrt(variance * 2 / dim) * (np.cos(np.dot(omega, x.T).T + b.T))
    return phi

class FunctionGenerator():
    def __init__(self, seed, lengthscale, variance, noise_var, X):
        self.seed = seed
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.X = X

    def gen(self,X_train,y_train,func_num):   #func_numは生成する関数の個数
        dim=1000
        random_state = check_random_state(self.seed)
        omega = (np.sqrt(1/(self.lengthscale**2)) * random_state.normal(size=(dim, X_train.shape[1])))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        large_phi = RFM(X_train, dim,omega,b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed,dim, self.noise_var)
        Theta.calc(large_phi, y_train)
        phi = RFM(self.X, dim, omega, b, self.variance)
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta,phi.T)
        return f