import numpy as np
import sys
import random
import math
from nptyping import NDArray
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

def func(x: NDArray[float]) -> NDArray[float]: #テスト用のforrester関数
    # 最大化したいので符号を反転
    return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)

class RBFKernel():
    def __init__(self, variance: float,length_scale: float):
        self.variance=variance
        self.length_scale = length_scale

    def __call__(self, X1: NDArray[float], X2: NDArray[float]) -> NDArray[float]:
        temp1= np.c_[np.sum(X1**2,axis=1)]
        temp2 = np.c_[np.sum(X2**2, axis=1)]
        norm = temp1 + temp2.T - 2*np.dot(X1,X2.T)
        return self.variance * np.exp(-norm / (2*self.length_scale**2))

def RFM(x:NDArray[float],dim:int,omega:NDArray[float],b:NDArray[float]) -> NDArray[float]:
    phi=np.sqrt(2/dim)*(np.cos(np.dot(omega,x.T))+b)
    return phi

def Theta(phi: NDArray[float], y: NDArray[float]) -> NDArray[float]:
    noise_var = 1.0e-4
    phi_T_phi = np.dot(phi, phi.T)
    A = phi_T_phi + noise_var * np.eye(phi_T_phi.shape[0])
    Ainv=np.linalg.inv(A)
    Ainv_phi_T = np.dot(Ainv, phi)
    mu = np.dot(Ainv_phi_T, y)
    var = noise_var * Ainv
    theta = np.random.multivariate_normal(mu.ravel(), var, y.shape[0])
    return theta

def plot(mu: NDArray[float], var: NDArray[float],X:NDArray[float],y:NDArray[float],X_train:NDArray[float],y_train:NDArray[float]):
    plt.rcParams["font.size"] = 13
    plt.subplot(1, 1, 1)
    plt.title("Gaussian Process")
    plt.plot(X.ravel(), y, "g--", label="true")
    plt.plot(X.ravel(), mu, "b", label="pred_mean")
    plt.fill_between(X.ravel(), (mu + 2 * np.sqrt(var)).ravel(), (mu - 2 * np.sqrt(var)).ravel(), alpha=0.3, color="blue")
    plt.plot(X_train.ravel(), y_train.ravel(), "ro", label="observed")
    plt.legend(loc="lower left",prop={'size': 8})
    plt.savefig("GP.pdf")
    plt.close()

def experiment(seed: int, initial_num: int, max_iter: int):
    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 200
    index_list = range(grid_num)
    X = np.c_[np.linspace(0, 1, grid_num)]
    y = func(X)
    
    #初期点の生成
    random.seed(seed)
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    #カーネル行列の作成
    kernel=RBFKernel(variance=5,length_scale=0.2)
    K = kernel(X_train, X_train)

    #観測誤差の固定
    noise_var = 1.0e-4

    # 精度(precision)行列の計算
    precision = np.linalg.inv(K + noise_var*np.eye(np.size(X_train)))

    #予測分布の導出
    k_test_train = kernel(X, X_train) #k*T
    kK_ = np.dot(k_test_train, precision)
    pred_mu = np.dot(kK_, y_train).ravel()
    pred_var = kernel(X, X) - np.dot(kK_, k_test_train.T)
    pred_var_diag = np.diag(pred_var)

    #初期データの結果のプロット
    plot(pred_mu, pred_var_diag, X, y, X_train, y_train)

    dim=100
    omega = np.c_[np.random.randn(dim)]  #標準正規分布の乱数
    b = np.c_[np.random.rand(dim) * 2 * np.pi] #[0,2π]の一様乱数

    #RFMから特徴量ベクトルφ(x)を取得
    phi_x = RFM(X_train, dim,omega,b)  #D=100とした。

    #パラメータΘの獲得
    theta = Theta(phi_x, y_train)

    #ブラックボックス関数fの近似を取得する。
    phi = RFM(X, dim, omega, b)
    f_x = np.dot(theta, phi)

    #fのプロット
    
    for i in range(10):
        plt.plot(X, f_x[i], "b", label="true")
    plt.savefig("RFM's_GP.pdf")
    plt.close()

    #y*の獲得
    y_star = f_x.max(axis=1)
    print(y_star)
    
def main():
    argv = sys.argv
    seed = 0
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    experiment(seed,initial_num,max_iter)
    
if __name__ == "__main__":
    main()


''' ガウス過程回帰モデル（MES）の作り方

1. 初期(訓練)データを用意 p81
2. 関数を用意
3. カーネル何使おうか？決める(今回はRBF)
4. カーネル行列の作成
5. RFMの作成
6. 関数fの近似
7. y*の獲得
5. 活性化関数の作成(MES)
5. 新しい入力点x'の出力y'を求める

'''