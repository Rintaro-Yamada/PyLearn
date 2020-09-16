import numpy as np
import sys
import random
import math
from nptyping import NDArray
import matplotlib.pyplot as plt
#unresolved import 'numpy'Python(unresolved-import)

def func(x: NDArray[float]) -> NDArray[float]:
    # 最大化したいので符号を反転
    return (-1) * (6 * x - 2) ** 2 * np.sin(12 * x - 4)

class RBFKernel():
    def __init__(self, variance: float,length_scale: float):
        self.variance=variance
        self.length_scale = length_scale

    def __call__(self, X1: NDArray[float], X2: NDArray[float]) -> NDArray[float]:
        temp1= np.c_[np.sum(X1**2,axis=1)]
        temp2 = np.c_[np.sum(X2**2, axis=1)]
        #この上の式よくわからん　temp1=np.dot(X1.T,X1)ではいけないの？
        norm = temp1 + temp2.T - 2*np.dot(X1,X2.T)
        return self.variance * np.exp(-norm/self.length_scale)

def experiment(seed: int, initial_num: int, max_iter: int):
    # 定義域は[0, 1] でgrid_num分割して候補点を生成
    grid_num = 200
    index_list = range(grid_num)
    X = np.c_[np.linspace(0, 1, grid_num)]
    y = func(X)
    
    random.seed(seed)
    #初期点の生成
    train_index = random.sample(index_list, initial_num)
    X_train = X[train_index]
    y_train = y[train_index]

    #カーネル行列の作成
    kernel=RBFKernel(variance=1,length_scale=1.0)
    K=kernel(X_train,X_train)

    #観測誤差の分散は適当に固定
    noise_var = 1.0e-4 

    # 精度(precision)行列の計算
    precision = np.linalg.inv(K + noise_var*np.eye(np.size(X_train)))

    #テストデータの作成
    random.seed(seed+1)
    train_index = random.sample(index_list, 5)
    X_test = X[train_index]
    
    k_test_train=kernel(xtest,xtrain)
    kK_=



def main():
    argv = sys.argv
    seed = 0
    initial_num = int(argv[1])
    max_iter = int(argv[2])
    experiment(seed,initial_num,max_iter)
    
if __name__ == "__main__":
    main()


''' ガウス過程回帰モデル　の　作り方

1. 初期データを用意 p81
2. 関数を用意
3. カーネル何使おうか？決める(RBF)
4. カーネル行列の作成
5. 新しい入力点x'の出力y'を求める

'''